# General libraries
import pandas as pd
import numpy as np
import warnings
import lightgbm
import gc
import os

# Plots
from matplotlib import pyplot as plt

# Sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression

# Forecaster
from lightgbm import LGBMRegressor

# Multiprocessing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

# Plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython import display
from time import sleep
from math import ceil

# Cuda
import torch

# Forecaster class

class Forecaster:
    def __init__(self, df_input):
        """
        Initialize the Predictor with the input dataframe.

        :param df_input: A pandas DataFrame containing the input data
        """
        # Create a copy of the input dataframe to avoid modifying the original data
        self.df = df_input.copy()

        # Initialize dictionaries to store models, feature importances, and best hyperparameters
        self.models = {}
        self.feature_importances = {}
        self.best_hyperparams = {}

    def remove_outliers(self, column, group_cols, lower_quantile=0.025, upper_quantile=0.975):
        """
        Cap the outliers in a specified column for each group defined by group_cols.
        """
        # Calculate lower and upper bounds for each group using transform
        lower_bounds = self.df.groupby(group_cols)[column].transform(lambda x: x.quantile(lower_quantile))
        upper_bounds = self.df.groupby(group_cols)[column].transform(lambda x: x.quantile(upper_quantile))

        # Cap the outliers using vectorized operations
        self.df[column] = np.clip(self.df[column], lower_bounds, upper_bounds)

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
            train_data = df[df['sample'] == 'train']
            test_data = df[df['sample'] == 'test']

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

    def prepare_features_for_selection(self, train_data, group_cols, numeric_features, target_col, sample_fraction=0.20):
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
        train_data_sampled = pd.merge(train_data, sampled_group_ids, on=group_cols, how='inner')

        # Fill NA with 0 for the selected data
        train_data_filled = train_data_sampled.fillna(0)

        # Prepare the feature and target data
        X_sampled = train_data_filled[group_cols + numeric_features]
        y_sampled = train_data_filled[target_col]

        return X_sampled, y_sampled

    def select_best_features(self, X, y, group_cols, n_best_features, n_groups=5000):
        """
        Select the best numeric features using mutual information regression for time series data,
        grouped by group_cols, using a subset of random groups.
        :param X: DataFrame of features
        :param y: Series of target values
        :param group_cols: List of columns to group by
        :param n_best_features: Number of best features to select
        :param n_groups: Number of random groups to use for feature selection
        :return: List of selected feature names
        """
        # Select only float64 columns
        numeric_features = X.select_dtypes(include=['float64']).columns

        # Concatenate X and y to ensure both are cleaned of NaN rows simultaneously
        data = pd.concat([X[group_cols], X[numeric_features], y], axis=1)

        # Function to compute MI scores for a group
        def compute_mi_scores(group):
            # Select features and target
            X_clean = group[numeric_features]
            y_clean = group[y.name]

            # Check if the lengths of X_clean and y_clean are the same
            if len(X_clean) == len(y_clean) and len(X_clean) > 0:
                return mutual_info_regression(X_clean, y_clean)
            return np.zeros(len(numeric_features))

        # Compute MI scores for each group
        grouped_mi_scores = data.groupby(group_cols).apply(compute_mi_scores)

        # Average MI scores across groups
        avg_mi_scores = grouped_mi_scores.mean()

        # Store the features and their average mutual information scores in a DataFrame
        feature_scores = pd.DataFrame({'feature': numeric_features, 'mi_score': avg_mi_scores})

        # Select the n_best_features
        best_features = feature_scores.nlargest(n_best_features, 'mi_score')['feature'].tolist()

        return best_features

    def load_default_params_and_distributions(self):
        """
        Load default parameters and hyperparameter distributions for the model.

        Returns:
        tuple: A tuple containing two dictionaries:
            - The first dictionary contains the default model parameters.
            - The second dictionary contains the default hyperparameter distributions for tuning.
        """
        default_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_iterations': 1000,
            'learning_rate': 0.01,
            'num_leaves': 32,
            'max_depth': 8,
            'verbosity': -1
        }

        default_param_distributions = {
            'objective': ['regression'],
            'boosting_type': ['gbdt'],
            'num_iterations': [500, 1000, 1500],
            'learning_rate': [0.01, 0.025, 0.05],
            'num_leaves': [16, 32, 48],
            'max_depth': [4, 8, 12]
        }

        return default_params, default_param_distributions

    def _tune_hyperparameters(self, X_train, y_train, param_distributions, search_method='halving', n_splits=4,
                              scoring='neg_root_mean_squared_error', n_iter=30, sample_weight=None):
        """
        Tune hyperparameters using the specified search method.

        :param X_train: Training feature data
        :param y_train: Training target data
        :param param_distributions: Dictionary of hyperparameter search space
        :param search_method: The search method to use ('grid', 'random', or 'halving')
        :param n_splits: Number of splits for TimeSeriesSplit (cross-validation)
        :param scoring: Scoring metric to use for optimization
        :param n_iter: Number of iterations for RandomizedSearchCV and HalvingRandomSearchCV
        :param sample_weight: Optional sample weights for training
        :return: The best estimator (model) found by the search and the best parameters
        """
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Set the device for LGBMRegressor
        if torch.cuda.is_available():
            base_model = LGBMRegressor(device_type='gpu', verbosity=-1)
        else:
            base_model = LGBMRegressor(verbosity=-1)

        # Choose the appropriate search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_distributions,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1
            )
        elif search_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                cv=tscv,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        elif search_method == 'halving':
            search = HalvingRandomSearchCV(
                estimator=base_model,
                param_distributions=param_distributions,
                max_resources=2500,
                cv=tscv,
                factor=3,
                scoring=scoring,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError("Invalid search_method. Choose 'grid', 'random', or 'halving'.")

        # Perform the hyperparameter search, passing sample weights if provided
        if sample_weight is not None:
            search.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_

    def train_model(self, df, cutoff, features, params, training_group, training_group_val, target_col,
                    use_weights=False, tune_hyperparameters=False, search_method='halving',
                    param_distributions=None, scoring='neg_root_mean_squared_error'):
        """
        Train the model for a specific cutoff and training group, and optionally perform hyperparameter tuning.

        :param df: Input data
        :param cutoff: The cutoff value for splitting the data
        :param features: List of feature names to use in the model
        :param params: Dictionary of model parameters to use if tuning is not performed
        :param training_group_val: The value of the training group to filter the data by
        :param target_col: The column to use as the target for prediction
        :param training_group: The column name to use for training groups
        :param use_weights: Whether to use the 'weight' column during training
        :param tune_hyperparameters: If True, will perform hyperparameter tuning
        :param search_method: The search method to use for hyperparameter tuning ('grid', 'random', or 'halving')
        :param param_distributions: Hyperparameter search space to be used if tuning is enabled
        :param scoring: Scoring metric to use for optimization
        :return: Modified DataFrame with predictions for the training and test sets
        """
        # Prepare the data
        X_train, y_train, X_test, test_idx, train_idx = self._prepare_data(df, features, target_col)

        # Extract sample weights if use_weights is enabled
        if use_weights:
            train_weights = df.loc[train_idx, 'train_weight'].values
        else:
            train_weights = None

        # Perform hyperparameter tuning if enabled
        if tune_hyperparameters and param_distributions:
            print(f"Tuning hyperparameters for cutoff: {cutoff}, training group: {training_group_val} ({training_group})")
            model, best_params = self._tune_hyperparameters(X_train, y_train, param_distributions,
                                                            search_method=search_method, scoring=scoring,
                                                            sample_weight=train_weights)
            self.models[(cutoff, training_group_val)] = model
            self.best_hyperparams[(cutoff, training_group_val)] = best_params
        else:
            # Check if CUDA is available
            if torch.cuda.is_available():
                model = LGBMRegressor(device_type='gpu', **params)
            else:
                model = LGBMRegressor(**params)

        # Fit the model with or without sample weights
        model.fit(X_train, y_train, sample_weight=train_weights)

        # Store model
        self.models[(cutoff, training_group_val)] = model

        # Store feature importances
        importances = model.feature_importances_
        self.feature_importances[(cutoff, training_group_val)] = importances

        # Make predictions
        predictions_train = model.predict(X_train)
        predictions_test = model.predict(X_test)

        # Store predictions in the dataframe
        df.loc[train_idx, 'prediction'] = predictions_train
        df.loc[test_idx, 'prediction'] = predictions_test

        # Return the modified DataFrame
        return df

    def process_cutoff(self, cutoff, features, params, target_col, training_group,
                       training_group_values, tune_hyperparameters, search_method,
                       param_distributions, scoring, best_features, n_best_features,
                       group_cols, baseline_col, use_guardrail, guardrail_limit,
                       use_weights):
        """
        Process cutoff function
        """
        try:
            print(f"Processing cutoff: {cutoff}")
            print("----------------------------------------------------------")

            # Create empty object for results
            results_cutoff = []

            # Filter for cutoff
            cutoff_df = self.df[self.df['cutoff'] == cutoff]

            # Perform feature selection for this cutoff if requested, using only training data
            if best_features:
                print(f"Selecting {n_best_features} best numeric features for cutoff {cutoff} using mutual information")

                # Filter scope for train data
                train_data = cutoff_df[cutoff_df['sample'] == 'train']
                if train_data.empty:
                    raise ValueError(f"No training data found for cutoff: {cutoff}")

                # Separate numeric and non-numeric features
                numeric_features = [col for col in features if col in train_data.columns and train_data[col].dtype in ['float64']]
                non_numeric_features = [col for col in features if col in train_data.columns and train_data[col].dtype not in ['float64']]

                # Use the helper function to prepare features for selection
                try:
                    X_sampled, y_sampled = self.prepare_features_for_selection(train_data, group_cols, numeric_features, target_col)
                except Exception as e:
                    print(f"Unexpected error during best feature selection sampling and preparation: {str(e)}")
                    raise

                # Show number of features before selection
                print(f"Total number of features before selection: {len(features)}")

                # Run selection function on numeric features only
                try:
                    selected_numeric_features = self.select_best_features(X_sampled, y_sampled, group_cols, n_best_features)
                except Exception as e:
                    print(f"Unexpected error during best feature selection: {str(e)}")
                    raise

                # Combine selected numeric features with all non-numeric features
                selected_features = selected_numeric_features + non_numeric_features

                # Show number of features after selection
                print(f"Total number of features after selection: {len(selected_features)}")

                # Assign new features
                features_to_use = selected_features
            else:
                # No feature selection
                print(f"No feature selection: Using all features provided")
                features_to_use = features

            # Iterate over training groups
            for training_group_val in training_group_values:
                print(f"Training and predicting for cutoff: {cutoff}, training group: {training_group_val}")

                # Filter scope
                cutoff_df_tg = cutoff_df[cutoff_df[training_group] == training_group_val]

                # Train model will save results
                try:
                    cutoff_df_results = self.train_model(cutoff_df_tg, cutoff, features_to_use, params, training_group, training_group_val, target_col,
                                                         use_weights, tune_hyperparameters, search_method, param_distributions, scoring)
                except Exception as e:
                    print(f"Error occurred while training model for cutoff {cutoff}, training group {training_group_val}: {str(e)}")
                    raise

                # Apply guardrail if specified
                if group_cols and baseline_col and use_guardrail:
                    print(f"Calculating guardrail for cutoff: {cutoff}, training group: {training_group_val}")

                    # Scope for guardrail
                    test_data = cutoff_df_results[cutoff_df_results['sample'] == 'test']

                    # Check if data is available
                    if test_data.empty:
                        print(f"Warning: No test data found for cutoff {cutoff}, training group {training_group_val}")
                    else:
                        try:
                            adjusted_data = self.calculate_guardrail(test_data, group_cols, baseline_col, guardrail_limit)
                            cutoff_df_results.loc[cutoff_df_results.index, 'prediction'] = adjusted_data['prediction']
                        except Exception as e:
                            print(f"Error occurred while calculating guardrail for cutoff {cutoff}, training group {training_group_val}: {str(e)}")
                            raise

                # Append the current cutoff_tg_results to results_cutoff
                results_cutoff.append(cutoff_df_results)

            # Concatenate all results
            results_cutoff = pd.concat(results_cutoff, axis=0, ignore_index=True)

            # Print dimensions
            print(f"Dimensions of results dataframe: {results_cutoff.shape}")

            return results_cutoff

        except Exception as e:
            print(f"Error occurred while processing cutoff {cutoff}: {str(e)}")
            raise

    def process_cutoff_wrapper(self, args):
        return self.process_cutoff(*args)

    def run_backtesting(self, group_cols=None, features=None, params=None, training_group='training_group',
                        target_col='sales', tune_hyperparameters=False, search_method='halving',
                        param_distributions=None, scoring='neg_mean_squared_log_error', best_features=None,
                        n_best_features=15, remove_outliers=False, outlier_column=None,
                        lower_quantile=0.025, upper_quantile=0.975, baseline_col=None,
                        use_guardrail=False, guardrail_limit=2.5, use_weights=False,
                        use_parallel=True, num_cpus=None):
        """
        Train and predict for all cutoff values and training groups in the dataset.
        """
        try:
            # Validate input
            if features is None:
                raise ValueError("Features list cannot be None. Pass columns with 'feature' in the name.")

            if group_cols is None:
                raise ValueError("group_cols cannot be None. Specify the columns used for grouping.")

            # Load default parameters and distributions if not provided
            default_params, default_param_distributions = self.load_default_params_and_distributions()

            # Assign default parameters
            if params is None:
                params = default_params

            if param_distributions is None:
                param_distributions = default_param_distributions

            # Remove outliers if specified
            if remove_outliers and outlier_column and group_cols:
                try:
                    new_group_cols = group_cols + ['cutoff']
                    self.remove_outliers(outlier_column, group_cols=new_group_cols, lower_quantile=lower_quantile, upper_quantile=upper_quantile)
                except Exception as e:
                    print(f"Error occurred while removing outliers: {str(e)}")
                    raise

            # Get unique cutoffs and training group values
            cutoffs = self.df['cutoff'].unique()
            print("Number of cutoffs detected: ", len(cutoffs))
            training_group_values = self.df[training_group].unique()

            # Check the number of cutoffs
            if len(cutoffs) == 0:
                raise ValueError("No cutoff values found in the dataset.")
            elif len(cutoffs) == 1:
                print("Only one cutoff value found. Processing without parallelization.")
                use_parallel = False

            # Ensure training group values are integers and sort them
            try:
                training_group_values = sorted([int(val) for val in training_group_values])
            except ValueError:
                raise ValueError(f"Training group values in column '{training_group}' must be convertible to integers.")

            # Check if CUDA is available
            if torch.cuda.is_available():
                print("CUDA is available: Using GPU")
            else:
                num_cpu = os.cpu_count()
                print(f"CUDA is not available, using CPU with {num_cpu} available cores")

            # Set the number of CPUs to use
            if num_cpus is None:
                num_cpu = os.cpu_count()
                num_cpus = num_cpu // 2
                print(f"Using {num_cpus} CPUs for parallel computing")
            else:
                num_cpus = num_cpu

            # Default guardrail
            if use_guardrail:
                self.df['guardrail'] = False

            # Use ProcessPoolExecutor to parallelize by cutoff if specified
            if use_parallel:
                print(f"Running predictions in parallel with {num_cpus} cores")
                print("----------------------------------------------------------")
                try:
                    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                        args_list = [(c, features, params, target_col, training_group,
                                      training_group_values, tune_hyperparameters, search_method,
                                      param_distributions, scoring, best_features, n_best_features,
                                      group_cols, baseline_col, use_guardrail, guardrail_limit,
                                      use_weights)
                                    for c in cutoffs]
                        results = list(executor.map(self.process_cutoff_wrapper, args_list))
                except Exception as e:
                    print(f"Error occurred during parallel processing: {str(e)}")
                    raise
            else:
                print(f"Running predictions sequentially")
                print("----------------------------------------------------------")
                results = [self.process_cutoff(c, features, params, target_col, training_group,
                                              training_group_values, tune_hyperparameters, search_method,
                                              param_distributions, scoring, best_features, n_best_features,
                                              group_cols, baseline_col, use_guardrail, guardrail_limit,
                                              use_weights)
                          for c in cutoffs]

            # Combine results
            if len(cutoffs) == 1:
                self.df = results[0]
            else:
                self.df = pd.concat(results, axis=0, ignore_index=True)

            # Completion message
            print("----------------------------------------------------------")
            print("Predictions completed for all cutoffs and training groups")

            return self.df

        except Exception as e:
            print(f"An unexpected error occurred in run_predictions: {str(e)}")
            raise

    def calculate_guardrail(self, data, group_cols, baseline_col, guardrail_limit=3):
        """
        Calculate and apply guardrails to predictions, and add an indicator for guardrail application.

        :param data: DataFrame containing the predictions and baseline values
        :param group_cols: List of columns to group by for guardrail calculation
        :param baseline_col: Column name for the baseline predictions
        :param guardrail_limit: Limit for the guardrail adjustment (default: 3)
        :return: DataFrame with adjusted predictions and guardrail indicator
        """
        # Ensure non-negative values for prediction and baseline columns
        data['prediction'] = data['prediction'].clip(lower=0)
        data[baseline_col] = data[baseline_col].clip(lower=0)

        def adjust_group(group):
            # Calculate sum of predictions and baseline
            sum_prediction = group['prediction'].sum()
            sum_baseline = group[baseline_col].sum()

            # Apply guardrail if condition is met
            if sum_prediction > guardrail_limit * sum_baseline or sum_prediction < sum_baseline / guardrail_limit:
                group['prediction'] = (group['prediction'] / 2) + (group[baseline_col] / 2)
                group['guardrail'] = True
            else:
                group['guardrail'] = False
            return group

        # Apply the adjustment to each group
        adjusted_data = data.groupby(group_cols, group_keys=False).apply(adjust_group)
        adjusted_data['guardrail'] = adjusted_data['guardrail'].astype(bool)

        return adjusted_data

    def plot_feature_importance(self):
        """
        Plot the average feature importance across all cutoffs and training groups.
        """
        if not self.feature_importances:
            print("No feature importances available. Make sure you've trained models first.")
            return

        # Initialize dictionaries to store cumulative importances and feature counts
        cumulative_importances = {}
        feature_counts = {}

        # Sum up importances across all cutoffs and training groups
        for (cutoff, training_group_val), importances in self.feature_importances.items():
            model_key = (cutoff, training_group_val)

            # Check if the model exists for this (cutoff, training_group_val)
            if model_key not in self.models:
                print(f"Model for {model_key} not found, skipping...")
                continue

            model = self.models[model_key]

            for idx, importance in enumerate(importances):
                # Get feature name and handle missing or misaligned feature indices
                try:
                    feature = model.feature_name_[idx]
                except IndexError:
                    print(f"Feature index {idx} out of range for model {model_key}, skipping...")
                    continue

                # Aggregate importance for this feature
                if feature not in cumulative_importances:
                    cumulative_importances[feature] = 0
                    feature_counts[feature] = 0
                cumulative_importances[feature] += importance
                feature_counts[feature] += 1

        # Calculate average importances
        avg_importances = {feature: imp / feature_counts[feature]
                          for feature, imp in cumulative_importances.items()}

        # Sort features by importance
        sorted_features = sorted(avg_importances.items(), key=lambda x: x[1], reverse=True)
        features = [item[0] for item in sorted_features]
        importances = [item[1] for item in sorted_features]

        # Plot
        plt.figure(figsize=(12, 8))
        plt.barh(features, importances, color='steelblue')
        plt.xlabel('Average Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title('Average Feature Importance Across Cutoffs and Training Groups', fontsize=16)
        plt.gca().invert_yaxis()  # Invert y-axis to show most important features at the top
        plt.tight_layout()
        plt.show()

    def get_feature_importance(self):
        """
        Retrieve the feature importances for each training group and cutoff.

        :return: Dictionary of feature importances for each training group and cutoff
        """
        return self.feature_importances

    def get_best_hyperparams(self):
        """
        Retrieve the best hyperparameters found for each training group and cutoff during hyperparameter tuning.

        :return: Dictionary of best hyperparameters for each training group and cutoff
        """
        return self.best_hyperparams