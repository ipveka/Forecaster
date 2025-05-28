# Forecaster API

This API provides a FastAPI wrapper for the Forecaster package, allowing you to run time series forecasting pipelines via HTTP requests without the need to save plots or results.

## Features

- Simple REST API interface to the Forecaster pipeline
- JSON input/output for easy integration with any client
- Support for all Forecaster parameters
- Returns predictions, metrics, and feature importance
- Minimal dependencies beyond the core Forecaster package

## Installation

Ensure you have all required dependencies:

```bash
pip install -r requirements.txt
```

## Running the API

Start the API server:

```bash
cd src
python api.py
```

This will start the server on http://localhost:8000. The API includes automatic documentation at http://localhost:8000/docs.

## API Endpoints

### `/forecast` (POST)

Main endpoint for running a forecasting pipeline.

**Input**: JSON payload with the following structure:
- `data`: Array of records (your time series data)
- `date_col`: Name of the date column
- `group_cols`: List of columns to group by
- `signal_cols`: List of signal columns to forecast
- Various optional parameters for customizing the forecasting pipeline

**Output**: JSON response with:
- `status`: Success or error status
- `execution_time`: Time taken to run the pipeline
- `data_shape`: Shape of the resulting dataset
- `metrics`: Evaluation metrics
- `data`: Complete forecast results
- `feature_importance`: Feature importance scores

### `/health` (GET)

Health check endpoint to verify the API is running.

## Query Examples

### Example 1: Basic Forecasting with Sample Data

```python
import requests
import pandas as pd
import json
import time

# Generate or load your data
df = pd.read_csv("your_data.csv")

# Convert DataFrame to list of records
data = df.to_dict(orient='records')

# Convert datetime objects to ISO format strings
for record in data:
    if isinstance(record['date'], pd.Timestamp):
        record['date'] = record['date'].isoformat()

# Prepare request payload
payload = {
    "data": data,
    "date_col": "date",
    "group_cols": ["product", "store"],
    "signal_cols": ["sales"],
    "target": "sales",
    "horizon": 12,
    "freq": "W",
    "n_cutoffs": 3,
    "complete_dataframe": True,
    "smoothing": False,
    "model": "LGBM",
    "use_parallel": False
}

# Make API request
response = requests.post("http://localhost:8000/forecast", json=payload)
result = response.json()

# Process results
forecast_df = pd.DataFrame(result['data'])
print(f"Forecast shape: {forecast_df.shape}")
print(f"Metrics: {result['metrics']}")
```

### Example 2: Custom Data with Different Parameters

```python
# Create or load custom data
dates = pd.date_range(start='2022-01-01', periods=100, freq='D')
custom_df = pd.DataFrame({
    'date': dates,
    'region': ['North'] * 50 + ['South'] * 50,
    'product_type': ['A'] * 25 + ['B'] * 25 + ['A'] * 25 + ['B'] * 25,
    'sales': np.random.normal(100, 20, 100) + np.sin(np.arange(100)/5) * 10
})

# Convert DataFrame to list of records
custom_data = custom_df.to_dict(orient='records')
for record in custom_data:
    if isinstance(record['date'], pd.Timestamp):
        record['date'] = record['date'].isoformat()

# Prepare request payload with different parameters
custom_payload = {
    "data": custom_data,
    "date_col": "date",
    "group_cols": ["region", "product_type"],
    "signal_cols": ["sales"],
    "target": "sales",
    "horizon": 7,  # 7-day forecast
    "freq": "D",   # Daily data
    "n_cutoffs": 2,
    "smoothing": True,  # Enable smoothing
    "model": "RF",  # Random Forest model
    "tune_hyperparameters": True,  # Enable hyperparameter tuning
    "use_feature_selection": True,
    "use_guardrail": True
}

# Make API request
response = requests.post("http://localhost:8000/forecast", json=custom_payload)
result = response.json()
```

### Example 3: Minimal Configuration

```python
# Prepare minimal request payload
minimal_payload = {
    "data": data,  # Your data as list of records
    "date_col": "date",
    "group_cols": ["product", "store"],
    "signal_cols": ["sales"]
    # All other parameters will use API defaults
}

# Make API request
response = requests.post("http://localhost:8000/forecast", json=minimal_payload)
result = response.json()
```

### Example 4: Health Check

```python
# Health check endpoint
HEALTH_URL = "http://localhost:8000/health"
response = requests.get(HEALTH_URL)
health_result = response.json()
print(f"API Status: {health_result['status']}")
print(f"Timestamp: {health_result['timestamp']}")
```

For more detailed examples, see the Jupyter notebook at `notebooks/forecaster_api_examples.ipynb` and the example client at `src/api_client_example.py`.

## Parameters

The API supports all parameters available in the Runner class:

### Required Parameters
- `data`: Your time series data
- `date_col`: Name of the date column
- `group_cols`: List of columns to group by
- `signal_cols`: List of signal columns to forecast

### Optional Parameters
- `target`: Target column (defaults to first signal column)
- `horizon`: Forecasting horizon (default: 4)
- `freq`: Data frequency ('D', 'W', 'M')
- `n_cutoffs`: Number of cutoffs for backtesting (default: 1)
- `model`: Model to use ('LGBM', 'RF', 'GBM', 'ADA', 'LR')
- And many more (see API documentation for full list)

## Performance Considerations

- The API processes data in memory, so be mindful of the size of your dataset
- For large datasets, consider using `use_parallel=True` to enable parallel processing
- The API returns the full dataset with predictions, which can be large for big datasets
