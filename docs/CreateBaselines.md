# CreateBaselines Class Documentation

This class provides methods to calculate different types of baselines for time series data. It includes implementations for moving average (MA), linear regression (LR), and LightGBM regression (LGBM) baselines. These methods help to create benchmark predictions for various signal columns, allowing for better comparison with actual values and aiding in model evaluation.

## 1. `create_ma_baseline`

- **Description**: Adds moving average (MA) baselines and feature baselines for each signal column to the test set.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame with columns such as 'client', 'warehouse', 'product', 'date', 'sales', 'price', 'filled_sales', 'filled_price', and 'sample'.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to use for calculating the baselines.
  - `window_size` (int): Size of the moving average window.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}` and `feature_baseline_{signal_col}` for each signal column.

## 2. `create_lr_baseline`

- **Description**: Adds linear regression (LR) baselines and feature baselines for each signal column to the test set. For the train set, stores the actual values in the baseline columns.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame with columns such as 'client', 'warehouse', 'product', 'date', 'sales', 'price', 'filled_sales', 'filled_price', and 'sample'.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to predict using linear regression.
  - `feature_cols` (list): List of feature columns to use for training the linear regression model.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}_lr` and `feature_baseline_{signal_col}_lr` for each signal column.

## 3. `create_lgbm_baseline`

- **Description**: Adds LightGBM regression baselines and feature baselines for each signal column to the test set. For the train set, stores the actual values in the baseline columns.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame.
  - `group_cols` (list): List of columns to group by (e.g., `['client', 'warehouse', 'product']`).
  - `date_col` (str): Name of the date column.
  - `signal_cols` (list): List of signal columns to predict using LightGBM.
  - `feature_cols` (list): List of feature columns to use for training the regression model.
- **Returns**: pd.DataFrame: DataFrame with additional columns: `baseline_{signal_col}_lgbm` and `feature_baseline_{signal_col}_lgbm` for each signal column.