## Class: `DataPreparation`

This class is responsible for preparing the data before the forecasting process. It handles completing missing data, detecting time series frequency, smoothing signals, creating backtesting datasets, and ensuring proper forecasting horizons. The class has been optimized for performance with vectorized operations, making it efficient for large datasets.

---

### 1. `run_data_preparation`

- **Description**: Main function that coordinates the entire data preparation pipeline.
- **Parameters**: Takes in the dataset (`df`), grouping columns, date column, target variable, forecast horizon, data frequency (now optional), and other options (e.g., smoothing, number of cutoffs).
- **Returns**: A prepared DataFrame ready for forecasting, with completed missing data, smoothed signals, and a backtesting dataset.
- **Recent Updates**: Now supports automatic frequency detection when `freq` parameter is not provided.

---

### 2. `detect_frequency`

- **Description**: Automatically detects the frequency of time series data based on the date column.
- **Parameters**: Input DataFrame (`df`) and date column.
- **Returns**: A pandas frequency string (e.g., 'D', 'W-SUN', 'M', 'Q', 'Y') representing the detected frequency.
- **Details**: Analyzes time differences between dates to determine daily, weekly, monthly, quarterly, or yearly patterns. For weekly data, it can detect the specific day of the week (e.g., 'W-MON' for Monday).

---

### 3. `complete`

- **Description**: Fills in missing dates for each group in the dataset to ensure continuous time series.
- **Parameters**: Input DataFrame (`df`), grouping columns, date column, and frequency.
- **Returns**: A DataFrame where all groups have a full date range, with missing rows filled with NaNs.
- **Performance**: Optimized with vectorized operations for faster processing of large datasets.

---

### 4. `smoothing`

- **Description**: Smooths signal columns by filling missing values using a rolling average.
- **Parameters**: DataFrame (`df`), group columns, date column, signal columns, and the window size for the moving average (`dp_window_size`).
- **Returns**: A DataFrame with additional columns for smoothed signals.
- **Performance**: Uses pandas' groupby.transform for vectorized operations, significantly improving performance with large datasets.
- **Recent Updates**: Parameter has been renamed from `ma_window_size` to `dp_window_size` for clarity and consistency across the framework.

---

### 5. `get_latest_n_dates`

- **Description**: Retrieves the latest `n` distinct dates from the dataset, considering only dates where the target is > 0 and not NaN.
- **Parameters**: Input DataFrame (`df`), date column, target column, and the number of cutoffs to retrieve.
- **Returns**: A series of the latest `n` distinct dates with valid target values.
- **Recent Updates**: Now filters to only include dates with valid target data to ensure meaningful cutoffs.

---

### 6. `get_first_dates_last_n_months`

- **Description**: Retrieves cutoff dates for backtesting, ensuring the latest valid date is always included. For n_cutoffs > 1, also includes the first date of previous months. All dates are filtered to only include dates where target is > 0 and not NaN.
- **Parameters**: Input DataFrame (`df`), date column, target column, and number of cutoffs (`n_cutoffs`).
- **Returns**: A list of cutoff dates, with the latest valid date always first (for forecasting), followed by first dates of previous months (for backtesting).
- **Recent Updates**: 
  - Now always includes the latest date with valid target as the primary cutoff for forecasting
  - Filters to only consider dates where target > 0 and not NaN
  - Prevents issues with forecast horizons extending beyond actual data availability

---

### 7. `create_backtesting_df`

- **Description**: Creates train/test splits based on specified cutoff dates for backtesting.
- **Parameters**: Input DataFrame (`df`), date column, and cutoff dates.
- **Returns**: A DataFrame with train/test splits for each cutoff date.
- **Performance**: Uses efficient cross-join operations instead of multiple DataFrame copies, reducing memory usage and improving speed.

---

### 8. `add_horizon_last_cutoff`

- **Description**: Ensures that for the latest cutoff, the forecast horizon is covered by adding any missing dates.
- **Parameters**: Input DataFrame (`df`), grouping columns, date column, forecast horizon, and frequency (`freq`).
- **Returns**: A DataFrame with the required horizon filled after the latest cutoff.
- **Performance**: Optimized implementation that avoids deprecated pandas features and handles edge cases more efficiently.
- **Updates**: Improved handling of missing values in the forecast horizon.

---

## Performance Optimizations

The DataPreparation class has been optimized for better performance with large datasets:

1. **Vectorized Operations**: Methods like `smoothing` and `complete` use pandas' vectorized operations instead of loops.

2. **Efficient Memory Usage**: The class minimizes creation of unnecessary DataFrame copies, especially in `create_backtesting_df`.

3. **Improved Handling of Group Operations**: Operations on grouped data use more efficient patterns, avoiding deprecated features.

4. **Automatic Frequency Detection**: The class can now automatically detect the frequency of time series data, removing the need for manual specification.

5. **Robust Error Handling**: Better handling of edge cases and missing values throughout the pipeline.

These optimizations significantly improve processing speed and memory efficiency when working with large time series datasets.
