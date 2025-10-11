# Standard library imports
import gc
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from math import ceil
from time import sleep

# Third-party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from IPython import display
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle

# ML library imports
import lightgbm
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.experimental import enable_halving_search_cv
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from statsmodels.tsa.seasonal import seasonal_decompose

# Local imports
from utils.parameters import (
    create_param_heuristics,
    estimator_map,
    hyperparam_dictionary,
    tune_hyperparameters,
)

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# Forecaster class
class Forecaster:
    # Init
    def __init__(self, df_input):
        """
        Initialize the Forecaster class.

        Parameters:
        df_input (pd.DataFrame): Input DataFrame
        """
        self.df = df_input
        self.models = {}
        self.feature_importances = {}
        self.best_hyperparams = {}
        self.heuristic_params = {}

    # Remove outliers
    def remove_outliers(
        self,
        column,
        group_cols,
        lower_quantile=0.025,
        upper_quantile=0.975,
        ts_decomposition=False,
    ):
        """
        Cap the outliers in a specified column for each group defined by group_cols.
        Optionally, use time series decomposition to detect outliers in the residuals.

        Parameters:
        - column: The column from which to remove outliers.
        - group_cols: The columns to group by for calculating quantiles.
        - lower_quantile: The lower quantile threshold for capping.
        - upper_quantile: The upper quantile threshold for capping.
        - ts_decomposition: If True, decompose the time series to find outliers in the residuals.
        """
        if ts_decomposition:
            # Ensure that the column is suitable for time series decomposition
            for group in self.df[group_cols].unique():
                group_data = self.df[self.df[group_cols].eq(group).all(axis=1)]

                # Check for sufficient data points for decomposition
                if len(group_data) < 2:
                    continue

                # Decompose the time series
                result = seasonal_decompose(
                    group_data[column], model="additive", period=1
                )
                residual = result.resid

                # Calculate lower and upper bounds for residuals
                lower_bounds = np.quantile(residual.dropna(), lower_quantile)
                upper_bounds = np.quantile(residual.dropna(), upper_quantile)

                # Determine outliers based on residuals
                outlier_mask = (residual < lower_bounds) | (residual > upper_bounds)

                # Replace the outliers in the original column based on the outlier mask
                self.df.loc[outlier_mask, column] = np.clip(
                    self.df.loc[outlier_mask, column], lower_bounds, upper_bounds
                )

        else:
            # Calculate lower and upper bounds for each group, excluding NaNs
            lower_bounds = self.df.groupby(group_cols)[column].transform(
                lambda x: x.dropna().quantile(lower_quantile)
            )
            upper_bounds = self.df.groupby(group_cols)[column].transform(
                lambda x: x.dropna().quantile(upper_quantile)
            )

            # Cap the outliers using vectorized operations
            self.df[column] = np.clip(self.df[column], lower_bounds, upper_bounds)

    # Prepare data
    def _prepare_data(self, df, features, target_col):
        """
        Prepare the training and testing data for a specific cutoff and training group.

        :param df: The input dataset
        :param features: List of feature columns to use in the model
        :param target_col: The column name for the target variable
        :return: X_train, y_train, X_test, test_idx, train_idx
        """
        try:
            # Filter data for training and testing
            train_data = df[df["sample"] == "train"]
            test_data = df[df["sample"] == "test"]

            # Separate features and target variable
            X_train = train_data[features]
            y_train = train_data[target_col]
            X_test = test_data[features]

            # Indices for tracking
            train_idx = train_data.index
            test_idx = test_data.index

            return X_train, y_train, X_test, test_idx, train_idx

        except Exception as e:
            print(f"Error occurred in prepare_data function: {str(e)}")
            raise

    # Prepare features for selection
    def prepare_features_for_selection(
        self, train_data, group_cols, numeric_features, target_col, sample_fraction=0.25
    ):
        """
        This function prepares the features for selection by:

        1. Dropping duplicates from the `group_cols` to get distinct group IDs.
        2. Sampling a specified fraction (default: 25%) of the distinct group IDs.
        3. Filtering the original dataset based on the sampled group IDs.
        4. Filling missing values (NA) with 0 in the selected data.
        5. Returning the sampled features (`X_sampled`) and the target variable (`y_sampled`).

        Args:
            train_data (pd.DataFrame): The original training dataset.
            group_cols (list): The columns representing the group IDs (used for grouping).
            numeric_features (list): The list of numeric features to select from.
            target_col (str): The target column for predictions.
            sample_fraction (float): Fraction of group IDs to sample.

        Returns:
            X_sampled (pd.DataFrame): DataFrame containing the sampled features.
            y_sampled (pd.Series): Series containing the sampled target variable.
        """

        # Find distinct group IDs by dropping duplicates based on group_cols
        unique_group_ids = train_data[group_cols].drop_duplicates()

        # Calculate the sample size
        sample_size = int(len(unique_group_ids) * sample_fraction)

        # Randomly sample the specified fraction of group IDs
        sampled_group_ids = unique_group_ids.sample(n=sample_size, random_state=42)

        # Filter the original dataset by performing a merge on the sampled group IDs
        train_data_sampled = pd.merge(
            train_data, sampled_group_ids, on=group_cols, how="inner"
        )

        # Fill NA with 0 for the selected data
        train_data_filled = train_data_sampled.fillna(0)

        # Prepare the feature and target data
        X_sampled = train_data_filled[group_cols + numeric_features]
        y_sampled = train_data_filled[target_col]

        return X_sampled, y_sampled

    # Select best features
    def select_best_features(self, X, y, group_cols, n_best_features):
        """
        Select the best numeric features using mutual information regression for time series data,
        grouped by group_cols, using a subset of random groups.
        :param X: DataFrame of features
        :param y: Series of target values
        :param group_cols: List of columns to group by
        :param n_best_features: Number of best features to select
        :return: List of selected feature names
        """
        try:
            # Select only float64 columns
            numeric_features = X.select_dtypes(include=["float64"]).columns

            if len(numeric_features) == 0:
                print("Warning: No float64 features found for feature selection.")
                return []

            # Validate group columns exist in X
            for col in group_cols:
                if col not in X.columns:
                    print(
                        f"Warning: Group column '{col}' not found in features. Skipping feature selection."
                    )
                    return numeric_features.tolist()[
                        : min(n_best_features, len(numeric_features))
                    ]

            # Concatenate X and y to ensure both are cleaned of NaN rows simultaneously
            data = pd.concat([X[group_cols], X[numeric_features], y], axis=1)

            # Remove any rows with NaN values in numeric features or target
            data = data.dropna(subset=[y.name] + numeric_features.tolist())

            if data.empty:
                print(
                    "Warning: No valid data for feature selection after removing NaN values."
                )
                return numeric_features.tolist()[
                    : min(n_best_features, len(numeric_features))
                ]

            # Function to compute MI scores for a group
            def compute_mi_scores(group):
                # Select features and target
                X_clean = group[numeric_features]
                y_clean = group[y.name]

                # Check if the lengths of X_clean and y_clean are the same and if they have at least 2 samples
                if len(X_clean) == len(y_clean) and len(X_clean) > 1:
                    try:
                        return mutual_info_regression(X_clean, y_clean)
                    except ValueError as e:
                        if "Expected n_neighbors < n_samples_fit" in str(e):
                            # If the error is due to n_neighbors being greater than n_samples_fit,
                            # set n_neighbors to 1 less than the number of samples
                            return mutual_info_regression(
                                X_clean, y_clean, n_neighbors=len(X_clean) - 1
                            )
                        else:
                            print(f"ValueError in mutual_info_regression: {str(e)}")
                            return np.zeros(len(numeric_features))
                    except Exception as e:
                        print(f"Unexpected error in compute_mi_scores: {str(e)}")
                        return np.zeros(len(numeric_features))
                else:
                    return np.zeros(len(numeric_features))

            # Compute MI scores for each group with error handling
            all_scores = []
            for name, group in data.groupby(group_cols):
                try:
                    scores = compute_mi_scores(group)
                    if scores is not None and len(scores) == len(numeric_features):
                        all_scores.append(scores)
                except Exception as e:
                    print(f"Error processing group {name}: {str(e)}")
                    continue

            # If no valid scores were computed, return all numeric features
            if not all_scores:
                print(
                    "Warning: Could not compute feature importance scores for any group."
                )
                return numeric_features.tolist()[
                    : min(n_best_features, len(numeric_features))
                ]

            # Convert to numpy array and compute the mean
            all_scores_array = np.array(all_scores)
            avg_mi_scores = np.mean(all_scores_array, axis=0)

            # Store the features and their average mutual information scores in a DataFrame
            feature_scores = pd.DataFrame(
                {"feature": numeric_features, "mi_score": avg_mi_scores}
            )

            # Select the n_best_features
            best_features = feature_scores.nlargest(n_best_features, "mi_score")[
                "feature"
            ].tolist()
            return best_features

        except Exception as e:
            print(f"Unexpected error during best feature selection: {str(e)}")
            # Return a subset of numeric features as a fallback
            if "numeric_features" in locals() and len(numeric_features) > 0:
                return numeric_features.tolist()[
                    : min(n_best_features, len(numeric_features))
                ]
            return []

    # Train model
    def train_model(
        self,
        df,
        cutoff,
        features,
        params,
        training_group,
        training_group_val,
        target_col,
        use_weights=False,
        tune_hyperparameters=False,
        search_method="halving",
        param_distributions=None,
        scoring="neg_root_mean_squared_error",
        n_iter=30,
        model="LGBM",
    ):
        """
        Train the model for a specific cutoff and training group, and optionally perform hyperparameter tuning.
        """
        # Prepare the data
        X_train, y_train, X_test, test_idx, train_idx = self._prepare_data(
            df, features, target_col
        )

        # Fill NA in y
        y_train = y_train.fillna(0)

        # Extract sample weights if use_weights is enabled
        if use_weights:
            train_weights = df.loc[train_idx, "train_weight"].values
        else:
            train_weights = None

        # Perform hyperparameter tuning if enabled
        if tune_hyperparameters:
            print(f"   ⚙️  Tuning hyperparameters...")
            print(f"      - Method: {search_method}")
            print(f"      - Scoring: {scoring}")
            # Tune hyperparams
            model, best_params = tune_hyperparameters(
                X_train,
                y_train,
                model=model,
                params=params,
                param_distributions=param_distributions,
                search_method=search_method,
                scoring=scoring,
                n_iter=n_iter,
                sample_weight=train_weights,
            )
            self.models[(cutoff, training_group_val)] = model
            self.best_hyperparams[(cutoff, training_group_val)] = best_params
        else:
            # Use heuristic parameters based on dataset characteristics
            print(f"   ⚙️  Using heuristic parameters (auto-configured)")
            model, heuristic_params = create_param_heuristics(X_train, model_type=model)

            # Store heuristic parameters
            self.heuristic_params[model] = heuristic_params

            # Log the heuristic parameters used (compact format)
            dataset_info = heuristic_params.get("dataset_info", {})
            print(f"      - Samples: {dataset_info.get('n_samples', 'N/A'):,}")
            print(f"      - Features: {dataset_info.get('n_features', 'N/A')}")
            
            # Show key parameters only
            key_params = ['n_estimators', 'learning_rate', 'max_depth']
            for param in key_params:
                if param in heuristic_params:
                    print(f"      - {param}: {heuristic_params[param]}")

            # Also store in best_hyperparams for consistency
            self.best_hyperparams[(cutoff, training_group_val)] = heuristic_params

        # Fit the model with or without sample weights
        model.fit(X_train, y_train, sample_weight=train_weights)

        # Store model
        self.models[(cutoff, training_group_val)] = model

        # Store feature importances if available
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
            self.feature_importances[(cutoff, training_group_val)] = importances

        # Make predictions
        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)

        # Store predictions in the dataframe
        df.loc[train_idx, "prediction"] = predictions_train
        df.loc[test_idx, "prediction"] = predictions_test

        return df

    # Process cutoff function
    def process_cutoff(
        self,
        cutoff,
        features,
        params,
        target_col,
        training_group,
        model,
        training_group_values,
        tune_hyperparameters,
        search_method,
        param_distributions,
        scoring,
        n_iter,
        best_features,
        n_best_features,
        group_cols,
        baseline_col,
        use_guardrail,
        guardrail_limit,
        use_weights,
        current_cutoff_idx,
        total_cutoffs,
    ):
        """
        Process cutoff function with progress updates
        """
        try:
            # Calculate and display cutoff progress percentage
            cutoff_progress = (current_cutoff_idx + 1) / total_cutoffs * 100
            print(f"\n" + "=" * 70)
            print(f"📅 CUTOFF {current_cutoff_idx + 1}/{total_cutoffs} ({cutoff_progress:.1f}%)")
            print(f"=" * 70)
            print(f"   • Date: {pd.to_datetime(cutoff).date()}")
            print(f"   • Training groups: {len(training_group_values)}")

            # Create empty object for results
            results_cutoff = []

            # Filter for cutoff
            cutoff_df = self.df[self.df["cutoff"] == cutoff]

            # Perform feature selection for this cutoff if requested, using only training data
            if best_features:
                print(f"\n🎯 Feature Selection:")
                print(f"   • Method: Mutual Information")
                print(f"   • Target features: {n_best_features}")

                # Filter scope for train data
                train_data = cutoff_df[cutoff_df["sample"] == "train"]
                if train_data.empty:
                    raise ValueError(f"No training data found for cutoff: {cutoff}")

                # Separate numeric and non-numeric features
                numeric_features = [
                    col
                    for col in features
                    if col in train_data.columns
                    and train_data[col].dtype in ["float64"]
                ]
                non_numeric_features = [
                    col
                    for col in features
                    if col in train_data.columns
                    and train_data[col].dtype not in ["float64"]
                ]

                # Use the helper function to prepare features for selection
                try:
                    X_sampled, y_sampled = self.prepare_features_for_selection(
                        train_data, group_cols, numeric_features, target_col
                    )
                except Exception as e:
                    print(
                        f"Unexpected error during best feature selection sampling and preparation: {str(e)}"
                    )
                    raise

                # Show number of features before selection
                print(f"   • Features before: {len(features)}")

                # Run selection function on numeric features only
                try:
                    selected_numeric_features = self.select_best_features(
                        X_sampled, y_sampled, group_cols, n_best_features
                    )
                except Exception as e:
                    print(f"Unexpected error during best feature selection: {str(e)}")
                    raise

                # Combine selected numeric features with all non-numeric features
                selected_features = selected_numeric_features + non_numeric_features

                # Show number of features after selection
                print(f"   • Features after: {len(selected_features)}")
                print(f"   ✓ Feature selection completed")

                # Assign new features
                features_to_use = selected_features
            else:
                # No feature selection
                print(f"\n🎯 Feature Selection: Disabled")
                print(f"   • Using all {len(features)} features provided")
                features_to_use = features

            # Iterate over training groups and track progress
            total_training_groups = len(training_group_values)
            for idx, training_group_val in enumerate(training_group_values):
                group_progress = (idx + 1) / total_training_groups * 100
                print(f"\n" + "-" * 70)
                print(f"🤖 Training Group {idx + 1}/{total_training_groups} ({group_progress:.1f}%)")
                print(f"   • Group ID: {training_group_val}")
                print(f"   • Model: {model}")

                # Filter scope
                cutoff_df_tg = cutoff_df[
                    cutoff_df[training_group] == training_group_val
                ]

                # Train model will save results
                try:
                    cutoff_df_results = self.train_model(
                        cutoff_df_tg,
                        cutoff,
                        features_to_use,
                        params,
                        training_group,
                        training_group_val,
                        target_col,
                        use_weights,
                        tune_hyperparameters,
                        search_method,
                        param_distributions,
                        scoring,
                        n_iter,
                        model,
                    )
                except Exception as e:
                    print(
                        f"Error occurred while training model for cutoff {cutoff}, training group {training_group_val}: {str(e)}"
                    )
                    raise

                # Apply guardrail if specified
                if group_cols and baseline_col and use_guardrail:
                    print(f"\n🛡️  Applying guardrail constraints...")

                    # Scope for guardrail
                    test_data = cutoff_df_results[cutoff_df_results["sample"] == "test"]

                    # Check if data is available
                    if test_data.empty:
                        print(
                            f"Warning: No test data found for cutoff {cutoff}, training group {training_group_val}"
                        )
                    else:
                        try:
                            adjusted_data = self.calculate_guardrail(
                                test_data, group_cols, baseline_col, guardrail_limit
                            )
                            # Update predictions with guardrail-adjusted values
                            test_indices = cutoff_df_results[cutoff_df_results["sample"] == "test"].index
                            cutoff_df_results.loc[test_indices, "model_prediction"] = adjusted_data["model_prediction"].values
                            cutoff_df_results.loc[test_indices, "prediction"] = adjusted_data["prediction"].values
                            cutoff_df_results.loc[test_indices, "guardrail"] = adjusted_data["guardrail"].values
                        except Exception as e:
                            print(
                                f"Error occurred while calculating guardrail for cutoff {cutoff}, training group {training_group_val}: {str(e)}"
                            )
                            raise

                # Append the current cutoff_tg_results to results_cutoff
                results_cutoff.append(cutoff_df_results)

            # Concatenate all results
            results_cutoff = pd.concat(results_cutoff, axis=0, ignore_index=True)

            return results_cutoff

        except Exception as e:
            print(f"Error occurred while processing cutoff {cutoff}: {str(e)}")
            raise

    # Wrapper for parallel computing
    def process_cutoff_wrapper(self, args):
        return self.process_cutoff(*args)

    # Run backtesting
    def run_backtesting(
        self,
        group_cols=None,
        features=None,
        params=None,
        training_group="training_group",
        target_col="sales",
        model="LGBM",
        tune_hyperparameters=False,
        search_method="halving",
        param_distributions=None,
        scoring="neg_mean_squared_log_error",
        n_iter=30,
        best_features=None,
        n_best_features=15,
        remove_outliers=False,
        outlier_column=None,
        lower_quantile=0.025,
        upper_quantile=0.975,
        ts_decomposition=False,
        baseline_col=None,
        use_guardrail=False,
        guardrail_limit=2.5,
        use_weights=False,
        use_parallel=True,
        num_cpus=None,
    ):
        """
        Train and predict for all cutoff values and training groups in the dataset.
        """
        # Print header
        print("\n" + "=" * 70)
        print("MODEL TRAINING & BACKTESTING")
        print("=" * 70)
        
        try:
            # Validate input
            if features is None:
                raise ValueError(
                    "Features list cannot be None. Pass columns with 'feature' in the name."
                )

            # Validate group columns
            if group_cols is None:
                raise ValueError(
                    "group_cols cannot be None. Specify the columns used for grouping."
                )

            # Log initial dataset info
            initial_rows = len(self.df)
            n_features = len(features)
            print(f"\n📊 Input Dataset:")
            print(f"   • Rows: {initial_rows:,}")
            print(f"   • Features: {n_features}")
            print(f"   • Target: '{target_col}'")
            print(f"   • Model: {model}")

            # Assign default parameters
            if params is None:
                print(f"\n⚙️  Using default model parameters")
                params = estimator_map
            else:
                print(f"\n⚙️  Using user-provided model parameters")

            if param_distributions is None:
                print(f"   • Using default hyperparameter search space")
                param_distributions = hyperparam_dictionary
            else:
                print(f"   • Using user-provided hyperparameter search space")

            # Remove outliers if specified
            if remove_outliers and outlier_column and group_cols:
                print(f"\n🔍 Removing outliers from '{outlier_column}'...")
                try:
                    new_group_cols = group_cols + ["cutoff"]
                    self.remove_outliers(
                        outlier_column,
                        group_cols=new_group_cols,
                        lower_quantile=lower_quantile,
                        upper_quantile=upper_quantile,
                        ts_decomposition=ts_decomposition,
                    )
                    print(f"   ✓ Outliers removed (quantiles: {lower_quantile}-{upper_quantile})")
                except Exception as e:
                    print(f"   ✗ Error occurred while removing outliers: {str(e)}")
                    raise
            else:
                print(f"\n⏭️  Skipping outlier removal")

            # Get unique cutoffs and training group values
            cutoffs = self.df["cutoff"].unique()
            n_cutoffs = len(cutoffs)
            print(f"\n📅 Backtesting Configuration:")
            print(f"   • Number of cutoffs: {n_cutoffs}")
            
            # Display cutoff dates
            if n_cutoffs <= 5:
                for i, cutoff in enumerate(sorted(cutoffs), 1):
                    print(f"      {i}. {pd.to_datetime(cutoff).date()}")
            else:
                sorted_cutoffs = sorted(cutoffs)
                print(f"      First: {pd.to_datetime(sorted_cutoffs[0]).date()}")
                print(f"      Last: {pd.to_datetime(sorted_cutoffs[-1]).date()}")

            # Handle training group
            if training_group is None or training_group not in self.df.columns:
                print(f"   • Training groups: 1 (dummy group - all data together)")
                self.df["training_group"] = 1
                training_group = "training_group"
                training_group_values = [1]
            else:
                training_group_values = self.df[training_group].unique()
                # Ensure training group values are integers
                try:
                    training_group_values = sorted([int(val) for val in training_group_values])
                    print(f"   • Training groups: {len(training_group_values)} ({training_group})")
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Training group values in column '{training_group}' must be convertible to integers."
                    )

            # Calculate total models to train
            total_models = n_cutoffs * len(training_group_values)
            print(f"   • Total models to train: {total_models}")

            # Check the number of cutoffs
            if len(cutoffs) == 0:
                raise ValueError("No cutoff values found in the dataset.")
            elif len(cutoffs) == 1:
                print(f"\n⚠️  Only one cutoff found - processing sequentially")
                use_parallel = False

            # Model configuration
            print(f"\n🤖 Model Configuration:")
            print(f"   • Algorithm: {model}")
            print(f"   • Hyperparameter tuning: {tune_hyperparameters}")
            if tune_hyperparameters:
                print(f"      - Search method: {search_method}")
                print(f"      - Scoring: {scoring}")
                print(f"      - Iterations: {n_iter}")
            print(f"   • Feature selection: {best_features is not None}")
            if best_features is not None:
                print(f"      - Best features: {n_best_features}")
            print(f"   • Sample weights: {use_weights}")
            print(f"   • Guardrails: {use_guardrail}")
            if use_guardrail:
                print(f"      - Limit: {guardrail_limit}x baseline")

            # Check if CUDA is available and get CPU count
            print(f"\n💻 Compute Resources:")
            if torch.cuda.is_available():
                print(f"   • GPU: Available (CUDA enabled)")
            else:
                print(f"   • GPU: Not available (using CPU only)")

            # Get the number of available CPU cores
            available_cpus = os.cpu_count()
            print(f"   • Available CPU cores: {available_cpus}")

            # Set the number of CPUs to use
            if num_cpus is None:
                # Default to half the available cores if not specified
                num_cpus = max(1, available_cpus // 2)
                print(f"   • Using: {num_cpus} cores (default: 50%)")
            else:
                # Ensure we don't exceed available cores
                num_cpus = min(num_cpus, available_cpus)
                print(f"   • Using: {num_cpus} cores (user-specified)")

            # Default guardrail
            if use_guardrail:
                self.df["guardrail"] = False

            # Use ProcessPoolExecutor to parallelize by cutoff if specified
            if use_parallel:
                print(f"\n🚀 Running predictions in PARALLEL mode")
                print("=" * 70)
                try:
                    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                        args_list = [
                            (
                                c,
                                features,
                                params,
                                target_col,
                                training_group,
                                model,
                                training_group_values,
                                tune_hyperparameters,
                                search_method,
                                param_distributions,
                                scoring,
                                n_iter,
                                best_features,
                                n_best_features,
                                group_cols,
                                baseline_col,
                                use_guardrail,
                                guardrail_limit,
                                use_weights,
                                idx,
                                len(cutoffs),
                            )
                            for idx, c in enumerate(cutoffs)
                        ]
                        results = list(
                            executor.map(self.process_cutoff_wrapper, args_list)
                        )
                except Exception as e:
                    print(f"\n❌ Error during parallel processing: {str(e)}")
                    raise
            else:
                print(f"\n🔄 Running predictions in SEQUENTIAL mode")
                print("=" * 70)
                results = [
                    self.process_cutoff(
                        c,
                        features,
                        params,
                        target_col,
                        training_group,
                        model,
                        training_group_values,
                        tune_hyperparameters,
                        search_method,
                        param_distributions,
                        scoring,
                        n_iter,
                        best_features,
                        n_best_features,
                        group_cols,
                        baseline_col,
                        use_guardrail,
                        guardrail_limit,
                        use_weights,
                        idx,
                        len(cutoffs),
                    )
                    for idx, c in enumerate(cutoffs)
                ]

            # Combine results
            print(f"\n🔗 Combining results from all cutoffs...")
            if len(cutoffs) == 1:
                self.df = results[0]
            else:
                self.df = pd.concat(results, axis=0, ignore_index=True)

            # Reset index
            self.df = self.df.reset_index(drop=True)

            # Calculate predictions statistics
            n_predictions = (self.df["sample"] == "test").sum()
            n_train = (self.df["sample"] == "train").sum()
            
            # Final summary
            print(f"\n" + "=" * 70)
            print(f"✅ BACKTESTING COMPLETED")
            print(f"=" * 70)
            print(f"   • Models trained: {total_models}")
            print(f"   • Training samples: {n_train:,}")
            print(f"   • Predictions generated: {n_predictions:,}")
            print(f"   • Final dataset shape: {self.df.shape[0]:,} rows × {self.df.shape[1]} columns")
            if hasattr(self, 'feature_importances') and self.feature_importances:
                print(f"   • Feature importances: Available for {len(self.feature_importances)} models")
            print("=" * 70 + "\n")

            return self.df

        except Exception as e:
            print(f"\n❌ Error in run_backtesting: {str(e)}")
            raise
    
    # Calculate Guardrail
    def calculate_guardrail(self, data, group_cols, baseline_col, guardrail_limit=2.5):
        """
        Apply a guardrail to model predictions to prevent extreme deviations from a baseline.

        For each group defined by `group_cols`, this function compares the sum of 
        predictions with the sum of baseline values. If the predictions exceed the 
        specified `guardrail_limit` multiplier of the baseline (too high or too low), 
        predictions are proportionally scaled back to the guardrail limit.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame containing at least 'prediction' and the baseline column.
        group_cols : list of str
            Columns to group by when applying the guardrail.
        baseline_col : str
            Column name of the baseline values to compare against.
        guardrail_limit : float, default 2.5
            Maximum allowed ratio between prediction sum and baseline sum before adjustment.

        Returns
        -------
        pd.DataFrame
            DataFrame with:
            - 'model_prediction': Original predictions from the model
            - 'prediction': Guardrail-adjusted predictions
            - 'guardrail': Boolean indicating whether guardrail was applied
        """
        # Work on a copy to avoid modifying original DataFrame
        data = data.copy()

        # Store original model predictions before any modifications
        data["model_prediction"] = data["prediction"].copy()

        # Ensure non-negative values for predictions and baseline
        data["prediction"] = data["prediction"].clip(lower=0)
        data[baseline_col] = data[baseline_col].clip(lower=0)
        
        # Replace NaN and inf values with 0
        data["prediction"] = data["prediction"].replace([np.inf, -np.inf], 0).fillna(0)
        data[baseline_col] = data[baseline_col].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Calculate group-level statistics using transform for efficiency
        group_sum_prediction = data.groupby(group_cols)["prediction"].transform("sum")
        group_sum_baseline = data.groupby(group_cols)[baseline_col].transform("sum")
        
        # Calculate the ratio, handling division by zero
        # If baseline is 0, set ratio to 1 (no adjustment needed)
        ratio = np.where(
            group_sum_baseline > 0,
            group_sum_prediction / group_sum_baseline,
            1.0
        )
        
        # Ensure ratio is never exactly 0 to avoid division by zero in scaling factor
        # Replace 0 with 1.0 (no scaling needed for zero predictions)
        ratio = np.where(ratio == 0, 1.0, ratio)
        
        # Determine if guardrail should be applied
        # Guardrail triggers if ratio > guardrail_limit OR ratio < 1/guardrail_limit
        upper_limit_exceeded = ratio > guardrail_limit
        lower_limit_exceeded = ratio < (1.0 / guardrail_limit)
        guardrail_triggered = upper_limit_exceeded | lower_limit_exceeded
        
        # Calculate scaling factor for adjustment
        # If upper limit exceeded: scale down to guardrail_limit
        # If lower limit exceeded: scale up to 1/guardrail_limit
        # Otherwise: no scaling (factor = 1)
        # Safe division since we've ensured ratio is never 0
        scaling_factor = np.where(
            upper_limit_exceeded,
            guardrail_limit / ratio,
            np.where(
                lower_limit_exceeded,
                (1.0 / guardrail_limit) / ratio,
                1.0
            )
        )
        
        # Apply proportional scaling to predictions
        # This maintains the relative distribution of predictions within each group
        data["prediction"] = data["prediction"] * scaling_factor
        
        # Add guardrail indicator column (True only where adjustment was actually applied)
        data["guardrail"] = guardrail_triggered
        
        # Ensure the guardrail column is boolean
        data["guardrail"] = data["guardrail"].astype(bool)
        
        # Log guardrail statistics with detailed information
        if guardrail_triggered.any():
            n_groups_adjusted = data[data["guardrail"]].groupby(group_cols).ngroups
            total_groups = data.groupby(group_cols).ngroups
            n_rows_adjusted = data["guardrail"].sum()
            total_rows = len(data)
            
            print(f"   Guardrail applied:")
            print(f"      - Groups affected: {n_groups_adjusted}/{total_groups}")
            print(f"      - Rows adjusted: {n_rows_adjusted}/{total_rows}")
            
            # Show adjustment statistics
            adjustment_pct = ((data.loc[data["guardrail"], "prediction"] / 
                              data.loc[data["guardrail"], "model_prediction"]) - 1) * 100
            avg_adjustment = adjustment_pct.mean()
            print(f"      - Average adjustment: {avg_adjustment:+.1f}%")
        
        return data

    # Plot feature importance
    def plot_feature_importance(self, top_n=15, show_std=True, color_scale=True):
        """
        Plot the average feature importance across all cutoffs and training groups.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display (default: 15)
        show_std : bool
            Whether to show standard deviation as error bars (default: True)
        color_scale : bool
            Whether to use color gradient based on importance (default: True)
        """
        if not self.feature_importances:
            print("No feature importances available. Make sure you've trained models first.")
            return
        
        # Initialize dictionaries to store importances for each feature
        feature_importances_dict = {}
        
        # Collect all importances for each feature
        for (cutoff, training_group_val), importances in self.feature_importances.items():
            model_key = (cutoff, training_group_val)
            
            if model_key not in self.models:
                print(f"Model for {model_key} not found, skipping...")
                continue
            
            model = self.models[model_key]
            
            for idx, importance in enumerate(importances):
                try:
                    feature = model.feature_name_[idx]
                except IndexError:
                    print(f"Feature index {idx} out of range for model {model_key}, skipping...")
                    continue
                
                if feature not in feature_importances_dict:
                    feature_importances_dict[feature] = []
                feature_importances_dict[feature].append(importance)
        
        # Calculate statistics
        feature_stats = {}
        for feature, imps in feature_importances_dict.items():
            feature_stats[feature] = {
                'mean': np.mean(imps),
                'std': np.std(imps),
                'count': len(imps)
            }
        
        # Sort by mean importance and get top_n
        sorted_features = sorted(feature_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
        sorted_features = sorted_features[:top_n]
        
        features = [item[0] for item in sorted_features]
        means = [item[1]['mean'] for item in sorted_features]
        stds = [item[1]['std'] for item in sorted_features]
        counts = [item[1]['count'] for item in sorted_features]
        
        # Create figure with better styling
        fig, ax = plt.subplots(figsize=(14, max(8, top_n * 0.5)))
        
        # Generate colors based on importance if color_scale is True
        if color_scale:
            norm_means = np.array(means) / max(means) if max(means) > 0 else np.array(means)
            colors = plt.cm.viridis(norm_means)
        else:
            colors = ['steelblue'] * len(features)
        
        # Create horizontal bar chart
        y_pos = np.arange(len(features))
        bars = ax.barh(y_pos, means, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        
        # Add error bars if requested
        if show_std:
            ax.errorbar(means, y_pos, xerr=stds, fmt='none', ecolor='darkred', 
                        alpha=0.6, capsize=4, capthick=2, linewidth=1.5)
        
        # Label style
        label_fontsize = 12
        label_fontweight = "bold"
        tick_fontsize = 11
        
        # Customize axes
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features, fontsize=tick_fontsize)
        ax.set_xlabel('Average Importance', fontsize=label_fontsize, fontweight=label_fontweight)
        ax.set_ylabel('Features', fontsize=label_fontsize, fontweight=label_fontweight)
        
        # Update title
        total_features = len(feature_stats)
        if total_features > top_n:
            title = f'Top {top_n} Features by Average Importance\n(out of {total_features} total features across {len(self.feature_importances)} models)'
        else:
            title = f'Average Feature Importance\n(across {len(self.feature_importances)} models)'
        ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
        
        # Add grid for better readability
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)
        ax.set_axisbelow(True)
        
        # Invert y-axis so highest importance is at top
        ax.invert_yaxis()
        
        # Add colorbar if using color scale
        if color_scale:
            sm = plt.cm.ScalarMappable(cmap='viridis', 
                                    norm=plt.Normalize(vmin=min(means), vmax=max(means)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, pad=0.02)
            cbar.set_label('Importance Value', rotation=270, labelpad=20,
                        fontsize=label_fontsize, fontweight=label_fontweight)
        
        plt.tight_layout()
        plt.show()

    # Get feature importance
    def get_feature_importance(self):
        """
        Retrieve the feature importances for each training group and cutoff.

        :return: Dictionary of feature importances for each training group and cutoff
        """
        return self.feature_importances

    # Get best hyperparameters
    def get_best_hyperparams(self):
        """
        Retrieve the best hyperparameters found for each training group and cutoff during hyperparameter tuning.

        :return: Dictionary of best hyperparameters for each training group and cutoff
        """
        return self.best_hyperparams
