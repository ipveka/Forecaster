# Forecaster Class Documentation

This class provides a structured approach to preparing time series data, selecting relevant features, and optimizing models for accurate forecasting. This file contains documentation for the most relevant functions.

## 1. `train_model`

- **Description**: Trains the model for a specific cutoff and training group, with optional hyperparameter tuning.
- **Parameters**:
  - `df` (pd.DataFrame): Input data.
  - `cutoff` (int): The cutoff value for splitting the data.
  - `features` (list): List of feature names for the model.
  - `params` (dict): Model parameters to use if tuning is not performed.
  - `training_group` (str): Column name for training groups.
  - `training_group_val` (int): Value of the training group to filter data.
  - `target_col` (str): Column used as the target for prediction.
  - `use_weights` (bool): Whether to use the 'weight' column during training.
  - `tune_hyperparameters` (bool): If True, will perform hyperparameter tuning.
  - `search_method` (str): The search method for hyperparameter tuning.
  - `param_distributions` (dict): Hyperparameter search space if tuning is enabled.
  - `scoring` (str): Scoring metric for optimization.
- **Returns**: 
  - pd.DataFrame: Modified DataFrame with predictions for the training and test sets.

## 2. `_tune_hyperparameters`

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
- **Returns**: 
  - `best_estimator`: The best estimator (model) found by the search.
  - `best_params`: The best hyperparameters.

## 3. `process_cutoff`

- **Description**: Processes a cutoff point, performing feature selection and training for each training group.
- **Parameters**:
  - `cutoff` (int): The cutoff value to process.
  - `features` (list): List of feature names to use in the model.
  - `params` (dict): Model parameters.
  - `target_col` (str): The column used as the target for prediction.
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

## 4. `run_backtesting`

- **Description**: Trains and predicts for all cutoff values and training groups in the dataset, with options for hyperparameter tuning and outlier removal.
- **Parameters**:
  - `group_cols` (list): Columns to group by during predictions.
  - `features` (list): List of feature names to use in the model.
  - `params` (dict): Model parameters.
  - `training_group` (str): Column name for training groups.
  - `target_col` (str): Column used as the target for prediction.
  - `tune_hyperparameters` (bool): If True, perform hyperparameter tuning.
  - `search_method` (str): The search method for hyperparameter tuning.
  - `param_distributions` (dict): Hyperparameter search space if tuning is enabled.
  - `scoring` (str): Scoring metric for optimization.
  - `best_features` (bool): If True, perform feature selection.
  - `n_best_features` (int): Number of best features to select.
  - `remove_outliers` (bool): If True, remove outliers from the dataset.
  - `outlier_column` (str): Column used to identify outliers.
  - `lower_quantile` (float): Lower quantile for outlier removal.
  - `upper_quantile` (float): Upper quantile for outlier removal.
  - `baseline_col` (str): Column for baseline comparison.
  - `use_guardrail` (bool): If True, apply guardrail limits to predictions.
  - `guardrail_limit` (float): Limit value for guardrail adjustments.
  - `use_weights` (bool): Whether to use weights during training.
  - `use_parallel` (bool): If True, run predictions in parallel.
  - `num_cpus` (int, optional): Number of CPUs to use for parallel processing.
- **Returns**: 
  - pd.DataFrame: Combined results of predictions for all cutoffs and training groups.
