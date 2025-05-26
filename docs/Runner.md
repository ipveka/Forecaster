## Class: `Runner`

The `Runner` class serves as the orchestrator for the entire forecasting pipeline. It coordinates the execution of all components (data preparation, feature engineering, baseline creation, forecasting, and evaluation) in a streamlined process, making it easy to run the complete pipeline with a single function call.

---

### 1. `__init__`

- **Description**: Initializes the Runner class with logging and verbosity options.
- **Parameters**: 
  - `verbose` (bool, optional): Controls the level of output during execution.
  - `log_level` (str, optional): Sets the logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR').
- **Details**: Sets up the logging configuration and initializes the runner state.

---

### 2. `run_pipeline`

- **Description**: Main function that executes the complete forecasting pipeline from data preparation to evaluation.
- **Parameters**: The function accepts numerous parameters organized into categories:
  
  **Required parameters**:
  - `df` (pd.DataFrame): Input DataFrame with time series data
  - `date_col` (str): Column containing dates
  - `group_cols` (list): Columns to group the data by
  - `signal_cols` (list): Columns containing signal data (e.g., sales, inventory)
  
  **Data preparation parameters**:
  - `target` (str, optional): Target column for forecasting (defaults to first signal column)
  - `horizon` (int, optional): Forecasting horizon length
  - `freq` (str, optional): Frequency of the data ('D', 'W', 'M')
  - `n_cutoffs` (int, optional): Number of cutoffs for backtesting
  - `complete_dataframe` (bool, optional): Whether to fill in missing dates
  - `smoothing` (bool, optional): Whether to apply smoothing
  - `dp_window_size` (int, optional): Window size for smoothing operations
  
  **Feature engineering parameters**:
  - `fe_window_size` (tuple, optional): Window sizes for moving averages and statistics
  - `lags` (tuple, optional): Lag values for creating lag features
  - `fill_lags` (bool, optional): Whether to fill forward lag values
  - `n_clusters` (int, optional): Number of groups for quantile clustering
  
  **Baseline parameters**:
  - `baseline_types` (list, optional): Types of baselines to create ('MA', 'LR', 'ML')
  - `bs_window_size` (int, optional): Window size for moving average baseline
  
  **Forecasting parameters**:
  - `model` (str, optional): Model to use ('LGBM', 'RF', 'GBM', 'ADA', 'LR')
  - `training_group_col` (str, optional): Column for training group segmentation
  - `tune_hyperparameters` (bool, optional): Whether to tune hyperparameters
  - `use_feature_selection` (bool, optional): Whether to use feature selection
  - `remove_outliers` (bool, optional): Whether to remove outliers in forecasting
  - `outlier_column` (str, optional): Column from which to remove outliers
  - `lower_quantile` (float, optional): Lower quantile for outlier removal
  - `upper_quantile` (float, optional): Upper quantile for outlier removal
  - `ts_decomposition` (bool, optional): Whether to use time series decomposition
  - `baseline_col` (str, optional): Baseline column to use for comparison
  - Many other forecasting configuration options
  
- **Returns**: A tuple containing:
  - The final DataFrame with forecasts
  - A DataFrame with evaluation metrics
  - A DataFrame with cutoff statistics
  
- **Recent Updates**: Now uses standardized parameter naming with clear prefixes (dp_, fe_, bs_) and automatically detects frequency-specific parameters based on the data frequency when not explicitly provided.

---

### 3. Pipeline Execution

The `run_pipeline` function executes the following steps in sequence:

1. **Data Preparation**: Prepares the dataset by handling missing values, smoothing signals, and creating a backtesting structure.
2. **Feature Engineering**: Creates features including moving averages, lags, date features, and other statistical features.
3. **Baseline Creation**: Establishes benchmark models for comparison (e.g., moving average).
4. **Forecasting**: Trains and runs the selected forecasting model, making predictions for each cutoff.
5. **Evaluation**: Calculates performance metrics and compares model results against baselines.

Each step is timed and logged, with detailed output to track progress and performance.

---

### 4. Frequency-Based Parameter Management

The Runner class now smartly manages parameters based on the detected frequency of the data:

- If a parameter is not explicitly provided, it retrieves the appropriate value from `get_frequency_params` based on the data frequency.
- This ensures optimal parameter settings for different time series frequencies (daily, weekly, monthly) without requiring manual configuration.
- The standardized parameter naming scheme (dp_window_size, fe_window_size, bs_window_size) makes it clear which component each parameter belongs to.

---

### Usage Example

```python
from utils.runner import Runner

# Initialize the runner
runner = Runner(verbose=True)

# Run the complete pipeline
results_df, metrics_df, cutoff_stats = runner.run_pipeline(
    df=my_data,
    date_col='date',
    group_cols=['product', 'store'],
    signal_cols=['sales', 'inventory'],
    target='sales',
    freq='W',  # Weekly data
    n_cutoffs=3,
    model='LGBM',
    baseline_types=['MA', 'LR']
)
```

The Runner class greatly simplifies the forecasting workflow, providing a clean interface for executing complex forecasting pipelines with sensible defaults and extensive customization options.
