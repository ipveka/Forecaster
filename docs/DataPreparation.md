## Class: `DataPreparation`

This class is responsible for preparing the data before the forecasting process. It handles completing missing data, smoothing signals, creating backtesting dataset, and ensuring proper forecasting horizons.

---

### 1. `run_data_preparation`

- **Description**: Main function that coordinates the entire data preparation pipeline.
- **Parameters**: Takes in the dataset (`df`), grouping columns, date column, target variable, forecast horizon, data frequency, and other options (e.g., smoothing, number of cutoffs).
- **Returns**: A prepared DataFrame ready for forecasting, with completed missing data, smoothed signals, and a backtesting dataset.

---

### 2. `complete_dataframe`

- **Description**: Fills in missing dates for each group in the dataset to ensure continuous time series.
- **Parameters**: Input DataFrame (`df`), grouping columns, and date column.
- **Returns**: A DataFrame where all groups have a full date range, with missing rows filled with NaNs.

---

### 3. `smoothing`

- **Description**: Smooths signal columns by filling missing values using a rolling average.
- **Parameters**: DataFrame (`df`), group columns, date column, signal columns, and the window size for the moving average.
- **Returns**: A DataFrame with additional columns for smoothed signals.

---

### 4. `get_latest_n_dates`

- **Description**: Retrieves the latest `n` distinct dates from the dataset.
- **Parameters**: Input DataFrame (`df`), date column, and the number of cutoffs to retrieve.
- **Returns**: A series of the latest `n` distinct dates.

---

### 5. `get_first_dates_last_n_months`

- **Description**: Retrieves the first date from each of the last `n` months.
- **Parameters**: Input DataFrame (`df`), date column, and number of months (`n_cutoffs`).
- **Returns**: A list of the first dates of the last `n` months.

---

### 6. `create_backtesting_df`

- **Description**: Creates train/test splits based on specified cutoff dates for backtesting.
- **Parameters**: Input DataFrame (`df`), date column, and cutoff dates.
- **Returns**: A DataFrame with train/test splits for each cutoff date.

---

### 7. `add_horizon_last_cutoff`

- **Description**: Ensures that for the latest cutoff, the forecast horizon is covered by adding any missing dates.
- **Parameters**: Input DataFrame (`df`), grouping columns, date column, forecast horizon, and frequency (`freq`).
- **Returns**: A DataFrame with the required horizon filled after the latest cutoff.
