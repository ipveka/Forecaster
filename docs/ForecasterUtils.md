## Module: `forecaster_utils`

This module contains utility functions that support the entire forecasting pipeline. It provides essential functionality for data generation, visualization, parameter management, and results handling.

---

### 1. `get_frequency_params`

- **Description**: Provides frequency-specific parameters for the forecasting pipeline based on the detected frequency of the data.
- **Parameters**: 
  - `freq` (str): The frequency of the data ('D' for daily, 'W' for weekly, 'M' for monthly).
- **Returns**: A dictionary containing parameters customized for the specified frequency:
  - `horizon`: Forecasting horizon length
  - `dp_window_size`: Window size for data preparation smoothing operations
  - `fe_window_size`: Window sizes for feature engineering
  - `bs_window_size`: Window size for baseline creation
  - `lags`: Lag values for creating lag features
  - `periods`: Number of periods for sample data generation
- **Recent Updates**: Parameter names have been standardized with clearer prefixes (dp_, fe_, bs_) to indicate which component they belong to.

---

### 2. `generate_sample_data`

- **Description**: Generates synthetic time series data for testing and demonstration purposes.
- **Parameters**:
  - `freq` (str): Frequency of the data to generate ('D', 'W', 'M')
  - `periods` (int, optional): Number of periods to generate
  - `start_date` (str, optional): Start date for the generated data
- **Returns**: A DataFrame containing synthetic time series data with product, store, date, sales and inventory columns.
- **Details**: Creates realistic patterns including trends, seasonality, and product-specific behaviors.

---

### 3. `visualize_data`

- **Description**: Creates visualizations of the input data to help understand patterns and distributions.
- **Parameters**:
  - `df` (pd.DataFrame): The DataFrame containing the data
  - `date_col` (str): Column name for the date
  - `group_cols` (list): Columns to group by
  - `signal_cols` (list): Columns to visualize
  - `output_path` (str, optional): Path to save the visualization
- **Returns**: None, but saves a visualization to the specified path.

---

### 4. `visualize_forecasts_by_cutoff`

- **Description**: Creates visualizations of forecasts separated by cutoff dates, showing actual vs. predicted values.
- **Parameters**:
  - `df` (pd.DataFrame): DataFrame with forecasts
  - `date_col` (str): Column name for the date
  - `group_cols` (list): Columns to group by
  - `target` (str): Target column name
  - `cutoffs` (list): List of cutoff dates
  - `output_path` (str, optional): Path to save the visualization
- **Returns**: None, but saves visualizations to the specified path.

---

### 5. `save_results`

- **Description**: Saves the forecasting results and metrics to CSV files.
- **Parameters**:
  - `df` (pd.DataFrame): DataFrame with forecasts
  - `metrics_df` (pd.DataFrame): DataFrame with metrics
  - `cutoff_stats` (pd.DataFrame): DataFrame with cutoff statistics
  - `output_dir` (str): Directory to save results
- **Returns**: None, but saves results to the specified directory.

---

### Other Utility Functions

The module also contains various helper functions for:
- Date handling and conversion
- Statistical calculations
- Data transformation
- Output formatting

These utilities form the backbone of the Forecaster framework, providing consistent functionality across different components of the pipeline.
