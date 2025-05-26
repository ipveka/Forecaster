# Forecaster Class Documentation

This class provides a structured approach to preparing time series data, selecting relevant features, training forecasting models, and generating predictions. It includes functionality for feature selection, hyperparameter tuning, outlier handling, and guardrail mechanisms to ensure prediction quality. This documentation covers the main functions and their parameters.

## 1. `run_backtesting`

- **Description**: Main wrapper function that trains and predicts for all cutoff values and training groups in the dataset. This is the primary entry point for forecasting operations, handling feature selection, outlier removal, guardrail application, and parallel processing.
- **Parameters**:
  - `group_cols` (list): Columns to group by during predictions (e.g., `['product', 'store']`).
  - `features` (list): List of feature names to use in the model.
  - `params` (dict, optional): Model parameters. If None, uses default internal estimator map.
  - `training_group` (str): Column name for training groups. Default is `'training_group'`.
  - `target_col` (str): Column used as the target for prediction. Default is `'sales'`.
  - `model` (str): The model to use for training. Default is `'LGBM'`. Options include `'LGBM'`, `'RF'`, `'GBM'`, `'ADA'`, `'LR'`.
  - `tune_hyperparameters` (bool): If True, perform hyperparameter tuning. Default is `False`.
  - `search_method` (str): The search method for hyperparameter tuning. Default is `'halving'`. Options include `'grid'`, `'random'`, `'halving'`.
  - `param_distributions` (dict, optional): Hyperparameter search space if tuning is enabled. If None, uses default internal hyperparameter dictionary.
  - `scoring` (str): Scoring metric for optimization. Default is `'neg_mean_squared_log_error'`.
  - `best_features` (bool/list): If True, perform feature selection. If a list, uses those features directly. Default is `None`.
  - `n_best_features` (int): Number of best features to select. Default is `15`.
  - `remove_outliers` (bool): If True, remove outliers from the dataset. Default is `False`.
  - `outlier_column` (str): Column used to identify outliers. Default is `None`.
  - `lower_quantile` (float): Lower quantile for outlier removal. Default is `0.025`.
  - `upper_quantile` (float): Upper quantile for outlier removal. Default is `0.975`.
  - `ts_decomposition` (bool): If True, remove outliers using time series decomposition. Default is `False`.
  - `baseline_col` (str): Column for baseline comparison. Default is `None`.
  - `use_guardrail` (bool): If True, apply guardrail limits to predictions. Default is `False`.
  - `guardrail_limit` (float): Limit value for guardrail adjustments. Default is `2.5`.
  - `use_weights` (bool): Whether to use weights during training. Default is `False`.
  - `use_parallel` (bool): If True, run predictions in parallel. Default is `True`.
  - `num_cpus` (int, optional): Number of CPUs to use for parallel processing. Default is half of available CPUs.
- **Returns**: 
  - pd.DataFrame: Combined results of predictions for all cutoffs and training groups.

## 2. `remove_outliers`

- **Description**: Caps outliers in a specified column for each group defined by group_cols, with options for using time series decomposition to detect outliers in residuals.
- **Parameters**:
  - `column` (str): The column from which to remove outliers.
  - `group_cols` (list): The columns to group by for calculating quantiles.
  - `lower_quantile` (float): The lower quantile threshold for capping. Default is `0.025`.
  - `upper_quantile` (float): The upper quantile threshold for capping. Default is `0.975`.
  - `ts_decomposition` (bool): If True, decompose the time series to find outliers in the residuals. Default is `False`.
- **Returns**: None (modifies the internal DataFrame in-place).

## 3. `train_model`

- **Description**: Trains the model for a specific cutoff and training group, with optional hyperparameter tuning.
- **Parameters**:
  - `df` (pd.DataFrame): Input data.
  - `cutoff` (int): The cutoff value for splitting the data.
  - `features` (list): List of feature names for the model.
  - `params` (dict): Model parameters to use if tuning is not performed.
  - `training_group` (str): Column name for training groups.
  - `training_group_val` (int): Value of the training group to filter data.
  - `target_col` (str): Column used as the target for prediction.
  - `model` (str, optional): The model to use for training. Default is `'LGBM'`.
  - `use_weights` (bool): Whether to use the 'weight' column during training.
  - `tune_hyperparameters` (bool): If True, will perform hyperparameter tuning.
  - `search_method` (str): The search method for hyperparameter tuning.
  - `param_distributions` (dict): Hyperparameter search space if tuning is enabled.
  - `scoring` (str): Scoring metric for optimization.
- **Returns**: 
  - pd.DataFrame: Modified DataFrame with predictions for the training and test sets.

## 4. `_tune_hyperparameters`

- **Description**: Tunes hyperparameters using the specified search method (grid, random, or halving).
- **Parameters**:
  - `X_train` (pd.DataFrame): Training feature data.
  - `y_train` (pd.Series): Training target data.
  - `param_distributions` (dict): Dictionary defining the hyperparameter search space.
  - `search_method` (str): The search method to use (`'grid'`, `'random'`, or `'halving'`).
  - `n_splits` (int): Number of splits for TimeSeriesSplit (cross-validation).
  - `scoring` (str): Scoring metric to optimize (default is `'neg_root_mean_squared_error'`).
  - `n_iter` (int): Number of iterations for RandomizedSearchCV and HalvingRandomSearchCV (default is 30).
  - `sample_weight` (array-like, optional): Sample weights for training.
  - `model` (str, optional): The model to tune. Default is `'LGBM'`.
- **Returns**: 
  - `best_estimator`: The best estimator (model) found by the search.
  - `best_params`: The best hyperparameters.

## 5. `process_cutoff`

- **Description**: Processes a cutoff point, performing feature selection and training for each training group.
- **Parameters**:
  - `cutoff` (int): The cutoff value to process.
  - `features` (list): List of feature names to use in the model.
  - `params` (dict): Model parameters.
  - `target_col` (str): The column used as the target for prediction.
  - `model` (str, optional): The model to use for training. Default is `'LGBM'`.
  - `training_group` (str): Column name for training groups.
  - `training_group_values` (list): Values of the training group to iterate over.
  - `tune_hyperparameters` (bool): If True, perform hyperparameter tuning.
  - `search_method` (str): The search method for tuning.
  - `param_distributions` (dict): Hyperparameter search space if tuning is enabled.
  - `scoring` (str): Scoring metric for optimization.
  - `best_features` (bool): If True, perform feature selection.
  - `n_best_features` (int): Number of best features to select.
  - `group_cols` (list): Columns to group by for guardrail calculations.
  - `baseline_col` (str): The column used for baseline comparison.
  - `use_guardrail` (bool): If True, apply guardrail limits to predictions.
  - `guardrail_limit` (float): Limit value for guardrail adjustments.
  - `use_weights` (bool): Whether to use weights during training.
- **Returns**: 
  - pd.DataFrame: Combined results for the cutoff and all training groups.

## 6. `select_best_features`

- **Description**: Selects the best numeric features using mutual information regression for time series data, grouped by group_cols.
- **Parameters**:
  - `X` (pd.DataFrame): DataFrame of features.
  - `y` (pd.Series): Series of target values.
  - `group_cols` (list): List of columns to group by.
  - `n_best_features` (int): Number of best features to select.
- **Returns**:
  - list: List of selected feature names.

## 7. `prepare_features_for_selection`

- **Description**: Prepares features for selection by sampling a fraction of distinct group IDs and extracting the corresponding features and targets.
- **Parameters**:
  - `train_data` (pd.DataFrame): The original training dataset.
  - `group_cols` (list): The columns representing the group IDs (used for grouping).
  - `numeric_features` (list): The list of numeric features to select from.
  - `target_col` (str): The target column for predictions.
  - `sample_fraction` (float): Fraction of group IDs to sample. Default is `0.25`.
- **Returns**:
  - X_sampled (pd.DataFrame): DataFrame containing the sampled features.
  - y_sampled (pd.Series): Series containing the sampled target variable.

## 8. `calculate_guardrail`

- **Description**: Calculates and applies guardrails to predictions, adding an indicator for guardrail application. This prevents predictions from deviating too far from baseline values.
- **Parameters**:
  - `data` (pd.DataFrame): DataFrame containing the predictions and baseline values.
  - `group_cols` (list): List of columns to group by for guardrail calculation.
  - `baseline_col` (str): Column name for the baseline predictions.
  - `guardrail_limit` (float): Limit for the guardrail adjustment. Default is `2.5`.
- **Returns**:
  - pd.DataFrame: DataFrame with adjusted predictions and guardrail indicator.

## 9. `plot_feature_importance`

- **Description**: Plots the average feature importance across all cutoffs and training groups, providing a visual representation of which features contribute most to the predictions.
- **Parameters**: None
- **Returns**: None (displays a plot)

## 10. `get_feature_importance`

- **Description**: Retrieves the feature importances for each training group and cutoff.
- **Parameters**: None
- **Returns**:
  - dict: Dictionary of feature importances for each training group and cutoff.

## 11. `get_best_hyperparams`

- **Description**: Retrieves the best hyperparameters found for each training group and cutoff during hyperparameter tuning.
- **Parameters**: None
- **Returns**:
  - dict: Dictionary of best hyperparameters for each training group and cutoff.

## Default Model Configurations

The `Forecaster` class uses the following default estimators and hyperparameter search spaces when the user doesn't provide custom configurations:

```python
estimator_map = {
    'LGBM': LGBMRegressor(n_jobs=-1, objective='regression', random_state=42),
    'RF': RandomForestRegressor(random_state=42),
    'GBM': GradientBoostingRegressor(random_state=42),
    'ADA': AdaBoostRegressor(random_state=42),
    'LR': LinearRegression()

hyperparam_dictionary = {
    'LGBM': {
        'learning_rate': [0.001, 0.01, 0.05, 0.1],
        'n_estimators': [500, 1000, 2000, 3000],
        'num_leaves': [16, 31, 64, 128],
        'max_depth': [4, 8, 16, 32],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'min_child_samples': [5, 10, 20],
        'reg_alpha': [0, 0.1, 1.0],
        'reg_lambda': [0, 0.1, 1.0],
        'boosting_type': ['gbdt', 'dart']
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
        'validation_fraction': [0.1, 0.2],
        'n_iter_no_change': [5, 10, 20]
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
}
```