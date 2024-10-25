## Class: `FeatureEngineering`

This class is designed to facilitate the creation of various features necessary for data analysis and modeling. It includes methods for generating encoded features, date features, moving averages, lag features, and more. These features can help improve the performance of machine learning models by providing additional insights from the data.

---

## 1. `run_feature_engineering`

- **Description**: Main function to prepare the data by sequentially calling internal functions. It processes the input DataFrame (`df`) by creating various features based on specified parameters.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_cols`: List of columns to group the data.
  - `date_col`: Column containing dates.
  - `target`: Target variable for analysis.
  - `horizon`: Forecasting horizon.
  - `freq`: Frequency of the data.
  - `window_sizes`: Window sizes for statistical calculations.
  - `lags`: Lag values for creating lag features.
  - `n_clusters`: Number of groups for quantile clustering.
  - `train_weight_type`: Type of weighting for training weights.
- **Returns**: Prepared DataFrame.

---

## 2. `create_encoded_features`

- **Description**: Adds label-encoded features for the specified categorical columns in the DataFrame.
- **Parameters**:
  - `df`: Input DataFrame.
  - `categorical_columns`: List of column names to be label encoded.
- **Returns**: Modified DataFrame with new label-encoded columns prefixed with 'feature_'.

---

## 3. `create_periods_feature`

- **Description**: Creates a feature that counts the number of periods since the first non-zero signal for each group based on the target variable.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `date_column`: Column containing dates.
  - `target_col`: Column used to determine when to start counting.
- **Returns**: DataFrame with new period-related features.

---

## 4. `create_date_features`

- **Description**: Creates date-related features from the date column, including year, quarter, month, and weeks/months until the next end of quarter and year.
- **Parameters**:
  - `df`: Input DataFrame.
  - `date_col`: Name of the date column.
  - `freq`: Frequency type ('W' for weekly, 'M' for monthly).
- **Returns**: DataFrame with new date-related feature columns.

## 5. `create_ma_features`

- **Description**: Calculates moving averages for specified signal columns, grouped by specified columns, for multiple window sizes.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `signal_columns`: List of columns to calculate moving averages on.
  - `window_sizes`: List of integers representing the window sizes for moving averages.
- **Returns**: DataFrame with new moving average columns.

---

## 6. `create_moving_stats`

- **Description**: Calculates moving minimum and maximum for specified signal columns, grouped by specified columns, for multiple window sizes.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `signal_columns`: List of columns to calculate moving min/max on.
  - `window_sizes`: List of integers representing the window sizes for moving min/max.
- **Returns**: DataFrame with new moving minimum and maximum columns.

---

## 7. `create_lag_features`

- **Description**: Creates lag features for signal columns within each group, preventing data leakage for future forecasts.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `date_col`: The name of the date column used for sorting.
  - `signal_columns`: List of signal columns to create lag features for.
  - `lags`: List of lag values to calculate.
  - `forecast_window`: Forecasting window to prevent leakage.
- **Returns**: DataFrame with additional lag feature columns.

## 8. `create_cov`

- **Description**: Calculates the coefficient of variation for multiple value columns within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns to calculate CV for.
- **Returns**: DataFrame with additional CV columns for each value column.

---

## 9. `create_distinct_combinations`

- **Description**: Calculates the number of distinct values for combinations of a lower-level group with other higher-level group columns.
- **Parameters**:
  - `df`: Input DataFrame.
  - `lower_level_group`: The lower-level group column.
  - `group_columns`: List of higher-level group columns.
- **Returns**: DataFrame with new columns for each combination's distinct counts.

---

## 10. `create_quantile_clusters`

- **Description**: Creates quantile clusters for multiple value columns based on their mean values within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns to create clusters for.
  - `n_groups`: Number of quantile groups to create.
- **Returns**: DataFrame with additional cluster columns for each value column.

---

## 11. `create_history_clusters`

- **Description**: Creates quantile clusters for multiple value columns based on their max values within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns for which to create clusters.
  - `n_groups`: Number of quantile groups to create (default is 4 for quartiles).
- **Returns**: DataFrame with additional cluster columns for each value column, joined back to the full DataFrame.

---

## 12. `create_intermittence_clusters`

- **Description**: Creates quantile clusters for each group based on the intermittence of multiple value columns, defined as the number of zero values divided by the total number of dates, excluding leading zeros.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns to evaluate for intermittence.
  - `n_groups`: Number of quantile groups to create for intermittence (default is 10).
- **Returns**: DataFrame with additional intermittence cluster columns for each value column, joined back to the full DataFrame.

---

## 13. `create_train_weights`

- **Description**: Creates a 'weight' column for each group in a DataFrame, giving more weight to recent observations. Weights are calculated only for rows where 'sample' is 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_cols`: List of columns to group by.
  - `feature_periods_col`: Column name for the feature periods (e.g., weeks since inception).
  - `train_weight_type`: The type of weighting to use ('exponential' or 'linear'). Defaults to 'exponential'.
- **Returns**: A copy of the input DataFrame with an added 'weight' column for 'train' samples.

