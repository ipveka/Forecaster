#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test module for the Forecaster API.
This module contains tests to verify the API functionality.
"""

import unittest
import requests
import pandas as pd
import numpy as np
import json
import sys
import os
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions
from utils.forecaster_utils import generate_sample_data

# Load environment variables
load_dotenv()

# Get API configuration from environment variables
API_HOST = os.getenv("FORECASTER_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("FORECASTER_API_PORT", 8000))
API_KEY = os.getenv("FORECASTER_API_KEY", "")

# Base URL for API
BASE_URL = f"http://{API_HOST}:{API_PORT}"

class TestForecasterAPI(unittest.TestCase):
    """Test cases for the Forecaster API."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Prepare headers with API key if available
        self.headers = {}
        if API_KEY:
            self.headers["X-API-Key"] = API_KEY
            
        # Generate sample data for tests
        self.sample_df = generate_sample_data(freq='W', periods=60)
        
        # Convert DataFrame to list of records
        self.data = self.sample_df.to_dict(orient='records')
        
        # Convert datetime objects to ISO format strings
        for record in self.data:
            if isinstance(record['date'], pd.Timestamp):
                record['date'] = record['date'].isoformat()
        
        # Basic payload for tests
        self.basic_payload = {
            "data": self.data,
            "date_col": "date",
            "group_cols": ["product", "store"],
            "signal_cols": ["sales"],
            "target": "sales",
            "horizon": 4,
            "freq": "W",
            "n_cutoffs": 1,
            "model": "LGBM",
            "use_parallel": False
        }
    
    def test_health_endpoint(self):
        """Test the health check endpoint."""
        print("Testing health endpoint...")
        response = requests.get(f"{BASE_URL}/health", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertIn("timestamp", data)
        print("Health endpoint test passed!")
    
    def test_lgbm_model(self):
        """Test the forecast endpoint with LGBM model."""
        print("Testing LGBM model...")
        # Make API request
        response = requests.post(f"{BASE_URL}/forecast", json=self.basic_payload, headers=self.headers)
        self.assertEqual(response.status_code, 200)
        
        # Verify response structure
        data = response.json()
        self.assertEqual(data["status"], "success")
        
        # Verify metrics
        self.assertIn("metrics", data)
        metrics = data["metrics"]
        self.assertIsInstance(metrics, dict)
        
        # Check if prediction column exists in results
        result_df = pd.DataFrame(data["data"])
        self.assertIn("prediction", result_df.columns)
        
        print("LGBM model test passed!")
        print(f"Execution time: {data['execution_time']:.2f} seconds")
        print(f"Data shape: {data['data_shape']['rows']} rows × {data['data_shape']['columns']} columns")
        
        # Print sample metrics
        if metrics:
            print("\nMetrics:")
            for model_name, model_metrics in metrics.items():
                print(f"  {model_name}:")
                for metric_name, metric_value in model_metrics.items():
                    print(f"    {metric_name}: {metric_value:.4f}")
        
        return result_df

if __name__ == "__main__":
    unittest.main()
