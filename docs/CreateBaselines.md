# CreateBaselines Class Documentation

This class provides methods to calculate different types of baselines for time series data. It includes implementations for moving average (MA), linear regression (LR), and LightGBM regression (LGBM) baselines. These methods help to create benchmark predictions for various signal columns, allowing for better comparison with actual values and aiding in model evaluation.

The class has been designed to work within the Forecaster pipeline and follows a consistent pattern where a main wrapper function orchestrates the various baseline creation methods.

## 1. `run_baselines`

- **Description**: Main function to prepare baselines by calling specified baseline functions in order. This wrapper function allows users to create multiple types of baselines in a single operation.
- **Parameters**:
  - `df` (pd.DataFrame): Input DataFrame containing the data.
  - `group_cols` (list): Columns to group the data (e.g., ['client', 'warehouse', 'product']).
  - `date_col` (str): Column containing dates.
  - `signal_cols` (list): List of signal columns to create baselines for.
  - `baseline_types` (list): List of baseline types to create ('MA' for moving average, 'LR' for linear regression, 'ML' for LightGBM). Default: ['MA', 'LR', 'ML'].
  - `window_size` (int): Window size for moving average baseline. Default: 13.
  - `feature_cols` (list): Feature columns for regression models (required for 'LR' and 'ML' baseline types). Default: None.
- **Returns**: pd.DataFrame: Prepared DataFrame with baseline columns added.

## 2. `create_ma_baseline`

- **Description**: Adds moving average (MA) baselines and feature baselines for each signal column to the test set.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame with columns such as 'client', 'warehouse', 'product', 'date', 'sales', 'price', 'filled_sales', 'filled_price', and 'sample'.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to use for calculating the baselines.
  - `window_size` (int): Size of the moving average window.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}` and `feature_baseline_{signal_col}` for each signal column.

## 3. `create_lr_baseline`

- **Description**: Adds linear regression (LR) baselines and feature baselines for each signal column to the test set. For the train set, stores the actual values in the baseline columns.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame with columns such as 'client', 'warehouse', 'product', 'date', 'sales', 'price', 'filled_sales', 'filled_price', and 'sample'.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to predict using linear regression.
  - `feature_cols` (list): List of feature columns to use for training the linear regression model.
  - `debug` (bool): Whether to print debug information. Default: False.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}_lr` and `feature_baseline_{signal_col}_lr` for each signal column.

## 4. `create_lgbm_baseline`

- **Description**: Adds LightGBM regression baselines and feature baselines for each signal column to the test set. For the train set, stores the actual values in the baseline columns.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to predict using LightGBM.
  - `feature_cols` (list): List of feature columns to use for training the regression model.
  - `debug` (bool): Whether to print debug information. Default: False.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}_lgbm` and `feature_baseline_{signal_col}_lgbm` for each signal column.