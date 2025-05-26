## Class: `FeatureEngineering`

This class is designed to facilitate the creation of various features necessary for data analysis and modeling. It includes methods for generating encoded features, date features, moving averages, lag features, and more. These features can help improve the performance of machine learning models by providing additional insights from the data.

The class has been optimized for performance, utilizing vectorized operations and efficient pandas methods to handle large datasets effectively.

---

## 1. `run_feature_engineering`

- **Description**: Main function to prepare the data by sequentially calling internal functions. It processes the input DataFrame (`df`) by creating various features based on specified parameters.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_cols`: List of columns to group the data.
  - `date_col`: Column containing dates.
  - `target`: Target variable for analysis.
  - `freq`: Frequency of the data (default: None). If None, frequency will be auto-detected.
  - `fe_window_size`: Window sizes for statistical calculations (default: (4, 13)).
  - `lags`: Lag values for creating lag features (default: (4, 13)).
  - `fill_lags`: Whether to fill forward lag values (default: False).
  - `n_clusters`: Number of groups for quantile clustering (default: 10).
- **Recent Updates**: Parameter has been renamed from `window_sizes` to `fe_window_size` for clarity and consistency across the framework.
- **Returns**: Prepared DataFrame with all generated features.
- **Optimization**: The method now automatically converts the date column to datetime format and detects frequency if not provided.

---

## 2. `detect_frequency`

- **Description**: Automatically detects the frequency of time series data based on the date column. This method analyzes the time differences between consecutive dates to determine the appropriate frequency string.
- **Parameters**:
  - `df`: Input DataFrame containing the time series data.
  - `date_col`: The name of the date column to analyze.
- **Returns**: A string representing the detected frequency (e.g., 'D' for daily, 'W-MON' for weekly on Monday, 'M' for monthly).

---

## 3. `create_encoded_features`

- **Description**: Adds label-encoded features for the specified categorical columns in the DataFrame.
- **Parameters**:
  - `df`: Input DataFrame.
  - `categorical_columns`: List of column names to be label encoded.
- **Returns**: Modified DataFrame with new label-encoded columns prefixed with 'feature_'.

---

## 4. `create_periods_feature`

- **Description**: Creates a feature that counts the number of periods since the first non-zero signal for each group based on the target variable.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `date_column`: Column containing dates.
  - `target_col`: Column used to determine when to start counting.
- **Returns**: DataFrame with new period-related features.

---

## 5. `create_date_features`

- **Description**: Creates date-related features from the date column, including year, quarter, month, and weeks/months until the next end of quarter and year.
- **Parameters**:
  - `df`: Input DataFrame.
  - `date_col`: Name of the date column.
  - `freq`: Frequency type ('W' for weekly, 'M' for monthly).
- **Returns**: DataFrame with new date-related feature columns.

## 6. `create_ma_features`

- **Description**: Calculates moving averages for specified signal columns, grouped by specified columns, for multiple window sizes.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by (can be a string for a single column).
  - `signal_columns`: List of columns to calculate moving averages on (can be a string for a single column).
  - `window_sizes`: List/tuple of integers representing the window sizes for moving averages (can be a single integer).
- **Returns**: DataFrame with new moving average columns.
- **Optimization**: This method has been optimized to perform more efficiently by minimizing redundant groupby operations and using vectorized operations.

---

## 7. `create_moving_stats`

- **Description**: Calculates moving minimum and maximum for specified signal columns, grouped by specified columns, for multiple window sizes.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by (can be a string for a single column).
  - `signal_columns`: List of columns to calculate moving min/max on (can be a string for a single column).
  - `window_sizes`: List/tuple of integers representing the window sizes for moving min/max (can be a single integer).
- **Returns**: DataFrame with new moving minimum and maximum columns.
- **Optimization**: This method has been optimized to perform more efficiently by grouping once per signal column instead of once per window size.

---

## 8. `create_lag_features`

- **Description**: Creates lag features for signal columns within each group, preventing data leakage for future forecasts by setting lags within the forecast window to None and limiting lag depth within the test set.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by (can be a string for a single column).
  - `date_col`: The name of the date column used for sorting.
  - `signal_columns`: List of signal columns to create lag features for (can be a string for a single column).
  - `lags`: List/tuple of lag values to calculate (can be a single integer).
  - `fill_lags`: Whether to fill forward lag values (default: False).
- **Returns**: DataFrame with additional lag feature columns.
- **Optimization**: Uses vectorized operations for row numbering and masking for better performance, especially with large datasets.

## 9. `create_cov`

- **Description**: Calculates the coefficient of variation for multiple value columns within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by (can be a string for a single column).
  - `value_columns`: List of value columns to calculate CV for (can be a string for a single column).
- **Returns**: DataFrame with additional CV columns for each value column.
- **Optimization**: Uses masks instead of creating separate DataFrames for better memory efficiency. Handles edge cases like division by zero and NaN values more robustly.

---

## 10. `create_distinct_combinations`

- **Description**: Calculates the number of distinct values for combinations of a lower-level group with other higher-level group columns.
- **Parameters**:
  - `df`: Input DataFrame.
  - `lower_level_group`: The lower-level group column.
  - `group_columns`: List of higher-level group columns.
- **Returns**: DataFrame with new columns for each combination's distinct counts.

---

## 11. `create_quantile_clusters`

- **Description**: Creates quantile clusters for multiple value columns based on their mean values within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns to create clusters for.
  - `n_groups`: Number of quantile groups to create.
- **Returns**: DataFrame with additional cluster columns for each value column.

---

## 12. `create_history_clusters`

- **Description**: Creates quantile clusters for multiple value columns based on their max values within each group, excluding zeros and only using rows where 'sample' equals 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns for which to create clusters.
  - `n_groups`: Number of quantile groups to create (default is 4 for quartiles).
- **Returns**: DataFrame with additional cluster columns for each value column, joined back to the full DataFrame.

---

## 13. `create_intermittence_clusters`

- **Description**: Creates quantile clusters for each group based on the intermittence of multiple value columns, defined as the number of zero values divided by the total number of dates, excluding leading zeros.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by.
  - `value_columns`: List of value columns to evaluate for intermittence.
  - `n_groups`: Number of quantile groups to create for intermittence (default is 10).
- **Returns**: DataFrame with additional intermittence cluster columns for each value column, joined back to the full DataFrame.

---

## 14. `create_train_weights`

- **Description**: Creates a 'train_weight' column for each group in a DataFrame, giving more weight to recent observations. Weights are calculated only for rows where 'sample' is 'train'.
- **Parameters**:
  - `df`: Input DataFrame.
  - `group_columns`: List of columns to group by (can be a string for a single column).
  - `feature_periods_col`: Column name for the feature periods (e.g., weeks since inception).
  - `train_weight_type`: The type of weighting to use ('exponential' or 'linear'). Defaults to 'linear'.
- **Returns**: A copy of the input DataFrame with an added 'train_weight' column for 'train' samples.
- **Optimization**: Avoids deprecated groupby-apply patterns for more efficient and future-proof implementation.

## 15. `create_fcst_lag_number`

- **Description**: Adds a `fcst_lag` column to the DataFrame that starts counting rows within each group from the first occurrence of `sample = 'test'`. The count starts at 1 for the first row where `sample` is `'test'`, and increments by 1 for subsequent rows. Rows before the first `'test'` sample in each group will have `fcst_lag` set to `NaN`.
  
- **Parameters**:
  - `df`: Input DataFrame containing data to process.
  - `group_columns`: List of columns to group by, such as `['client', 'warehouse', 'product', 'cutoff']`.
  - `date_col`: Column used to order rows within each group. Defaults to `'date'`.
  - `sample_col`: Column name that identifies the target sample, such as `'sample'`.
  - `target_sample`: Value within `sample_col` to start counting from, usually `'test'`. Defaults to `'test'`.
  
- **Returns**: A copy of the input DataFrame with an added `fcst_lag` column, containing sequential counts for `sample = 'test'` rows in each group.