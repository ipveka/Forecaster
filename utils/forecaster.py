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

# Models
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LinearRegression

# Statsmodels
from statsmodels.tsa.seasonal import seasonal_decompose

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

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Define estimator map
estimator_map = {
    'LGBM': LGBMRegressor(n_jobs=-1, objective='regression', random_state=42, verbose=-1),
    'RF': RandomForestRegressor(random_state=42, verbose=-1),
    'GBM': GradientBoostingRegressor(random_state=42, verbose=-1),
    'ADA': AdaBoostRegressor(random_state=42),
    'LR': LinearRegression()
}

# Define parameter dictionary
hyperparam_dictionary = {
    'LGBM': {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [500, 1000, 2000],
        'num_leaves': [15, 31, 64],
        'max_depth': [4, 8, 12],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'reg_alpha': [0.0, 0.1, 0.5],
        'reg_lambda': [0.0, 0.1, 0.5]
    },
    'RF': {
        'n_estimators': [100, 500, 1000, 1500],
        'max_depth': [8, 16, 32, 64, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None],
        'criterion': ['squared_error', 'absolute_error'],
        'bootstrap': [True, False],
        'max_samples': [0.5, 0.7, 0.9]
    },
    'GBM': {
        'n_estimators': [100, 300, 500, 1000],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.6, 0.8, 1.0],
        'max_features': ['sqrt', 'log2', None],
        'validation_fraction': [0.1, 0.2]
    },
    'ADA': {
        'n_estimators': [50, 100, 200, 500],
        'learning_rate': [0.01, 0.05, 0.1, 0.3],
        'algorithm': ['SAMME', 'SAMME.R'],
        'random_state': [42]
    },
    'LR': {
        'alpha': [0.001, 0.01, 0.1, 1.0, 10, 100, 500],
        'fit_intercept': [True, False],
        'solver': ['svd', 'cholesky', 'lsqr', 'sag'],
        'max_iter': [1000],
        'normalize': [True, False],
        'tol': [1e-4, 1e-3]
    }
}

# Forecaster class
class Forecaster:
    # Init
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

    # Remove outliers
    def remove_outliers(self, column, group_cols, lower_quantile=0.025, upper_quantile=0.975, ts_decomposition=False):
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
                result = seasonal_decompose(group_data[column], model='additive', period=1)  # Adjust period as needed
                residual = result.resid

                # Calculate lower and upper bounds for residuals
                lower_bounds = np.quantile(residual.dropna(), lower_quantile)
                upper_bounds = np.quantile(residual.dropna(), upper_quantile)

                # Determine outliers based on residuals
                outlier_mask = (residual < lower_bounds) | (residual > upper_bounds)

                # Replace the outliers in the original column based on the outlier mask
                self.df.loc[outlier_mask, column] = np.clip(self.df.loc[outlier_mask, column], lower_bounds, upper_bounds)

        else:
            # Calculate lower and upper bounds for each group, excluding NaNs
            lower_bounds = self.df.groupby(group_cols)[column].transform(lambda x: x.dropna().quantile(lower_quantile))
            upper_bounds = self.df.groupby(group_cols)[column].transform(lambda x: x.dropna().quantile(upper_quantile))

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
    
    # Prepare features for selection
    def prepare_features_for_selection(self, train_data, group_cols, numeric_features, target_col, sample_fraction=0.25):
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
        # Select only float64 columns
        numeric_features = X.select_dtypes(include=['float64']).columns

        # Concatenate X and y to ensure both are cleaned of NaN rows simultaneously
        data = pd.concat([X[group_cols], X[numeric_features], y], axis=1)

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
                        return mutual_info_regression(X_clean, y_clean, n_neighbors=len(X_clean) - 1)
                    else:
                        raise e
            else:
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

    # Tune hyperparameters
    def _tune_hyperparameters(self, X_train, y_train, model='LGBM', params=None, param_distributions=None, search_method='halving', n_splits=4,
                              scoring='neg_root_mean_squared_error', n_iter=30, sample_weight=None):
        """
        Tune hyperparameters using the specified search method.
        """
        # Set up cross-validation
        tscv = TimeSeriesSplit(n_splits=n_splits)

        # Select the estimator and parameter distribution
        base_model = params[model]
        param_dist = param_distributions[model]

        # Print the selected base model and parameter distribution
        print("Selected Base Model:", base_model)
        print("Parameter Distribution:", param_dist)

        # Choose the search method
        if search_method == 'grid':
            search = GridSearchCV(
                estimator=base_model,
                param_grid=param_dist,
                cv=tscv,
                scoring=scoring,
                error_score='raise',
                n_jobs=-1,
                verbose=1
            )
        elif search_method == 'random':
            search = RandomizedSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                n_iter=n_iter,
                cv=tscv,
                scoring=scoring,
                error_score='raise',
                n_jobs=-1,
                verbose=1,
                random_state=42,
            )
        elif search_method == 'halving':
            search = HalvingRandomSearchCV(
                estimator=base_model,
                param_distributions=param_dist,
                max_resources=3000,
                aggressive_elimination=False,
                return_train_score=False,
                refit=True,
                cv=tscv,
                factor=3,
                scoring=scoring,
                error_score='raise',
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
        else:
            raise ValueError("Invalid search_method. Choose 'grid', 'random', or 'halving'.")

        # Perform the search, including sample weights if provided
        if sample_weight is not None:
            search.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_
    

    # Count and report NA
    def calculate_nan_inf_percentage(self, X, y):
        # Total number of rows
        total_rows_X = X.shape[0]
        total_rows_y = y.shape[0]

        # Check for NaNs
        nan_rows_X = np.isnan(X).any(axis=1)
        nan_rows_y = np.isnan(y)
        nan_count_X = np.sum(nan_rows_X)
        nan_count_y = np.sum(nan_rows_y)

        # Check for infinite values
        inf_rows_X = np.isinf(X).any(axis=1)
        inf_rows_y = np.isinf(y)
        inf_count_X = np.sum(inf_rows_X)
        inf_count_y = np.sum(inf_rows_y)

        # Calculate percentages
        nan_percentage_X = (nan_count_X / total_rows_X) * 100
        nan_percentage_y = (nan_count_y / total_rows_y) * 100
        inf_percentage_X = (inf_count_X / total_rows_X) * 100
        inf_percentage_y = (inf_count_y / total_rows_y) * 100

        # Print results
        print(f"NaNs in X_train: {nan_percentage_X:.2f}% of rows contain NaNs")
        print(f"NaNs in y_train: {nan_percentage_y:.2f}% of rows contain NaNs")
        print(f"Infinite values in X_train: {inf_percentage_X:.2f}% of rows contain infinite values")
        print(f"Infinite values in y_train: {inf_percentage_y:.2f}% of rows contain infinite values")

    # Train model
    def train_model(self, df, cutoff, features, params, training_group, training_group_val, target_col,
                    use_weights=False, tune_hyperparameters=False, search_method='halving',
                    param_distributions=None, scoring='neg_root_mean_squared_error', n_iter=30, 
                    model='LGBM'):
        """
        Train the model for a specific cutoff and training group, and optionally perform hyperparameter tuning.
        """
        # Prepare the data
        X_train, y_train, X_test, test_idx, train_idx = self._prepare_data(df, features, target_col)

        # Report NA and Infs
        self.calculate_nan_inf_percentage(X_train, y_train)

        # Fill NA in y
        y_train = y_train.fillna(0)

        # Extract sample weights if use_weights is enabled
        if use_weights:
            train_weights = df.loc[train_idx, 'train_weight'].values
        else:
            train_weights = None

        # Perform hyperparameter tuning if enabled
        if tune_hyperparameters:
            print(f"Tuning hyperparameters for cutoff: {cutoff}, training group: {training_group_val} ({training_group})")
            # Tune hyperparams
            model, best_params = self._tune_hyperparameters(
                X_train, y_train, model=model, params=params, param_distributions=param_distributions,
                search_method=search_method, scoring=scoring, n_iter=n_iter,
                sample_weight=train_weights)
            self.models[(cutoff, training_group_val)] = model
            self.best_hyperparams[(cutoff, training_group_val)] = best_params
        else:
            # Use specified model from estimator_map
            model = params[model]

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
        df.loc[train_idx, 'prediction'] = predictions_train
        df.loc[test_idx, 'prediction'] = predictions_test

        return df

    # Process cutoff function
    def process_cutoff(self, cutoff, features, params, target_col, training_group, model,
                       training_group_values, tune_hyperparameters, search_method,
                       param_distributions, scoring, n_iter, best_features, n_best_features,
                       group_cols, baseline_col, use_guardrail, guardrail_limit,
                       use_weights, current_cutoff_idx, total_cutoffs):
        """
        Process cutoff function with progress updates
        """
        try:
            # Calculate and display cutoff progress percentage
            cutoff_progress = (current_cutoff_idx + 1) / total_cutoffs * 100
            print(f"\nProcessing cutoff {current_cutoff_idx + 1}/{total_cutoffs} ({cutoff_progress:.2f}%) - Cutoff: {cutoff}")
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

            # Iterate over training groups and track progress
            total_training_groups = len(training_group_values)
            for idx, training_group_val in enumerate(training_group_values):
                group_progress = (idx + 1) / total_training_groups * 100
                print(f"Training and predicting for cutoff: {cutoff}, training group: {training_group_val} ({group_progress:.2f}% of groups in cutoff)")

                # Filter scope
                cutoff_df_tg = cutoff_df[cutoff_df[training_group] == training_group_val]

                # Train model will save results
                try:
                    cutoff_df_results = self.train_model(cutoff_df_tg, cutoff, features_to_use, params, training_group, training_group_val, target_col,
                                                         use_weights, tune_hyperparameters, search_method, param_distributions, scoring, n_iter, model)
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

            return results_cutoff

        except Exception as e:
            print(f"Error occurred while processing cutoff {cutoff}: {str(e)}")
            raise

    # Wrapper for parallel computing
    def process_cutoff_wrapper(self, args):
        return self.process_cutoff(*args)

    # Run backtesting
    def run_backtesting(self, group_cols=None, features=None, params=None, training_group='training_group',
                        target_col='sales', model='LGBM', tune_hyperparameters=False, search_method='halving',
                        param_distributions=None, scoring='neg_mean_squared_log_error', n_iter=30, best_features=None,
                        n_best_features=15, remove_outliers=False, outlier_column=None, lower_quantile=0.025, 
                        upper_quantile=0.975, ts_decomposition=False, baseline_col=None,
                        use_guardrail=False, guardrail_limit=2.5, use_weights=False,
                        use_parallel=True, num_cpus=None):
        """
        Train and predict for all cutoff values and training groups in the dataset.
        """
        # Print init
        print("Starting backtesting")
        try:
            # Validate input
            if features is None:
                raise ValueError("Features list cannot be None. Pass columns with 'feature' in the name.")

            # Validate group columns
            if group_cols is None:
                raise ValueError("group_cols cannot be None. Specify the columns used for grouping.")

            # Assign default parameters
            if params is None:
                print("User did not provide parameter dictionary, using internal method")
                params = estimator_map

            if param_distributions is None:
                print("User did not provide hyperparameter dictionary, using internal method")
                param_distributions = hyperparam_dictionary

            # Remove outliers if specified
            if remove_outliers and outlier_column and group_cols:
                try:
                    new_group_cols = group_cols + ['cutoff']
                    self.remove_outliers(outlier_column, group_cols=new_group_cols, lower_quantile=lower_quantile, upper_quantile=upper_quantile, 
                                         ts_decomposition=ts_decomposition)
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

            # Show selected model
            print("Predictions will be generated with model:", model)

            # Use ProcessPoolExecutor to parallelize by cutoff if specified
            if use_parallel:
                print(f"Running predictions in parallel with {num_cpus} cores")
                print("----------------------------------------------------------")
                try:
                    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
                        args_list = [(c, features, params, target_col, training_group, model,
                                      training_group_values, tune_hyperparameters, search_method,
                                      param_distributions, scoring, n_iter, best_features, n_best_features,
                                      group_cols, baseline_col, use_guardrail, guardrail_limit,
                                      use_weights, idx, len(cutoffs)) for idx, c in enumerate(cutoffs)]
                        results = list(executor.map(self.process_cutoff_wrapper, args_list))
                except Exception as e:
                    print(f"Error occurred during parallel processing: {str(e)}")
                    raise
            else:
                print(f"Running predictions sequentially")
                print("----------------------------------------------------------")
                results = [self.process_cutoff(c, features, params, target_col, training_group, model,
                                              training_group_values, tune_hyperparameters, search_method,
                                              param_distributions, scoring, n_iter, best_features, n_best_features,
                                              group_cols, baseline_col, use_guardrail, guardrail_limit,
                                              use_weights, idx, len(cutoffs)) for idx, c in enumerate(cutoffs)]

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
    
    # Calculate guardrail
    def calculate_guardrail(self, data, group_cols, baseline_col, guardrail_limit=2.5):
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

    # Plot feature importance
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
        plt.gca().invert_yaxis()
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