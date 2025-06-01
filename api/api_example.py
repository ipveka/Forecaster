#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Example client for the Forecaster API.
This script demonstrates how to call the API with sample data.
"""

import json
import os
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get API configuration from environment variables
API_HOST = os.getenv("FORECASTER_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("FORECASTER_API_PORT", 8000))
API_KEY = os.getenv("FORECASTER_API_KEY", "")

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.forecaster_utils import generate_sample_data


def main():
    """Main function to demonstrate API usage."""
    # API endpoint
    API_URL = f"http://{API_HOST}:{API_PORT}/forecast"

    # Generate sample data
    print("Generating sample data...")
    df = generate_sample_data(freq="W", periods=160)

    # Convert DataFrame to list of records
    data = df.to_dict(orient="records")

    # Convert datetime objects to ISO format strings
    for record in data:
        if isinstance(record["date"], pd.Timestamp):
            record["date"] = record["date"].isoformat()

    # Prepare request payload
    payload = {
        "data": data,
        "date_col": "date",
        "group_cols": ["product", "store"],
        "signal_cols": ["sales"],
        "target": "sales",
        "horizon": 13,
        "freq": "W",
        "n_cutoffs": 3,
        "complete_dataframe": True,
        "smoothing": False,
        "dp_window_size": 13,
        "fe_window_size": [4, 13],
        "bs_window_size": 13,
        "lags": [13, 52],
        "fill_lags": True,
        "baseline_types": ["MA"],
        "model": "LGBM",
        "tune_hyperparameters": False,
        "use_lags": True,
        "use_guardrail": False,
        "use_parallel": False,
    }

    # Prepare headers with API key if available
    headers = {}
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    # Make API request
    print("\nSending request to API...")
    start_time = time.time()

    try:
        response = requests.post(API_URL, json=payload, headers=headers)
        response.raise_for_status()

        # Process response
        result = response.json()

        # Calculate request time
        request_time = time.time() - start_time

        # Print results
        print("\n" + "=" * 80)
        print("FORECASTING API RESULTS")
        print("=" * 80)
        print(f"Status: {result['status']}")
        print(f"API request time: {request_time:.2f} seconds")
        print(f"Server execution time: {result['execution_time']:.2f} seconds")
        print(
            f"Data shape: {result['data_shape']['rows']} rows × {result['data_shape']['columns']} columns"
        )

        # Print metrics if available
        if result["metrics"]:
            print("\nMetrics:")
            for model, metrics in result["metrics"].items():
                print(f"  {model}:")
                for metric_name, metric_value in metrics.items():
                    print(f"    {metric_name}: {metric_value:.4f}")

        # Convert results back to DataFrame for analysis
        result_df = pd.DataFrame(result["data"])

        # Print sample of results
        print("\nSample of results (first 5 rows):")
        print(result_df.head())

        print("\n" + "=" * 80)
        print("API REQUEST COMPLETED SUCCESSFULLY")
        print("=" * 80)

    except requests.exceptions.RequestException as e:
        print(f"Error making API request: {str(e)}")
        if hasattr(e, "response") and e.response is not None:
            print(f"Response content: {e.response.text}")


if __name__ == "__main__":
    main()
