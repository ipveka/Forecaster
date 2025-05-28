#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Simplified test script for the Forecaster Runner class with different frequencies.
This script only verifies that the forecasting pipeline works for each frequency.
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import logging
import tempfile
import shutil
from datetime import datetime

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import utility functions and classes
from utils.runner import Runner
from utils.forecaster_utils import generate_sample_data, get_frequency_params

# Configure minimal logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def test_frequency(freq, model='LGBM'):
    """
    Test the forecasting pipeline with a specific frequency.
    
    Parameters:
    -----------
    freq : str
        Frequency to test ('D', 'W', or 'M')
    model : str, default='LGBM'
        Model to use for testing
    
    Returns:
    --------
    bool
        True if the test passed, False otherwise
    """
    print(f"\nTesting {freq} frequency with {model} model...")
    start_time = time.time()
    
    try:
        # Get frequency-specific parameters
        params = get_frequency_params(freq)
        
        # Use smaller dataset for faster testing
        if freq == 'D':
            periods = 365  # 1 year of daily data
        elif freq == 'W':
            periods = 52   # 1 year of weekly data
        else:  # Monthly
            periods = 12   # 1 year of monthly data
        
        # Generate sample data
        df = generate_sample_data(freq=freq, periods=periods)
        
        # Create a temporary directory for outputs
        temp_dir = tempfile.mkdtemp()
        
        try:
            # Configure the Runner
            runner = Runner(verbose=False)  # Reduce verbosity
            
            # Define basic parameters
            date_col = 'date'
            group_cols = ['product', 'store']
            signal_cols = ['sales']
            
            # Run the pipeline with minimal output
            result_df = runner.run_pipeline(
                # Required parameters
                df=df,
                date_col=date_col,
                group_cols=group_cols,
                signal_cols=signal_cols,
                
                # Data preparation parameters
                target='sales',
                horizon=params['horizon'],
                freq=freq,
                n_cutoffs=2,  # Reduce cutoffs for faster testing
                complete_dataframe=True,
                smoothing=False,
                dp_window_size=params['dp_window_size'],
                
                # Feature engineering parameters
                fe_window_size=params['fe_window_size'],
                lags=params['lags'],
                fill_lags=True,
                
                # Baseline parameters
                baseline_types=['MA'],
                bs_window_size=params['bs_window_size'],
                
                # Forecaster parameters
                model=model,
                training_group='training_group',
                tune_hyperparameters=False,
                use_feature_selection=False,
                use_lags=True,
                n_best_features=20,
                use_guardrail=False,
                use_parallel=False
            )
            
            # Check if we have predictions
            has_predictions = 'prediction' in result_df.columns
            
            # Get metrics
            metrics = runner.metrics
            
            # Extract RMSE values
            baseline_rmse = metrics['RMSE']['baseline_sales_ma_13'] if metrics and 'RMSE' in metrics and 'baseline_sales_ma_13' in metrics['RMSE'] else None
            prediction_rmse = metrics['RMSE']['prediction'] if metrics and 'RMSE' in metrics and 'prediction' in metrics['RMSE'] else None
            
            # Calculate improvement
            if baseline_rmse and prediction_rmse:
                improvement = ((baseline_rmse - prediction_rmse) / baseline_rmse * 100)
            else:
                improvement = None
            
            # Print minimal results
            execution_time = time.time() - start_time
            print(f"✅ {freq} frequency test passed in {execution_time:.2f} seconds")
            print(f"   Data shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
            if improvement:
                print(f"   Performance: {improvement:.1f}% improvement over baseline")
            
            return True
            
        finally:
            # Clean up temporary directory
            shutil.rmtree(temp_dir)
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ {freq} frequency test failed in {execution_time:.2f} seconds")
        print(f"   Error: {str(e)}")
        return False

def main():
    """
    Run tests for all frequencies with LGBM model.
    """
    print("\n" + "=" * 60)
    print("TESTING FORECASTER WITH DIFFERENT FREQUENCIES")
    print("=" * 60)
    
    # Start timing
    overall_start_time = time.time()
    
    # Track results
    results = {}
    
    # Test daily frequency
    results['D'] = test_frequency('D', 'LGBM')
    
    # Test weekly frequency
    results['W'] = test_frequency('W', 'LGBM')
    
    # Test monthly frequency
    results['M'] = test_frequency('M', 'LGBM')
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    all_passed = all(results.values())
    for freq, passed in results.items():
        status = "✅ Passed" if passed else "❌ Failed"
        print(f"{freq} Frequency: {status}")
    
    print("\nOverall Result: " + ("✅ All tests passed!" if all_passed else "❌ Some tests failed!"))
    print(f"Total execution time: {time.time() - overall_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
