#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Runner script for the Forecaster package.
This script demonstrates how to use the Runner class with different data frequencies.
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import argparse
import time

# Now import from absolute paths
from src.utils.forecaster_utils import (
    generate_sample_data,
    visualize_data,
    visualize_forecasts_by_cutoff,
    save_results,
    get_frequency_params
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Create output directory if it doesn't exist
OUTPUTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Main functon
def main():
    """Main function to run the Forecaster pipeline."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run the Forecaster pipeline')
    parser.add_argument('--freq', type=str, default='W', help='Data frequency: D (daily), W (weekly), or M (monthly)')
    parser.add_argument('--n_cutoffs', type=int, default=1, help='Number of cutoffs for backtesting')
    parser.add_argument('--tune', action='store_true', help='Enable hyperparameter tuning')
    parser.add_argument('--model', type=str, default='LGBM', choices=['LGBM', 'RF', 'GBM', 'ADA', 'LR'], help='Model to use for forecasting')
    args = parser.parse_args()
    
    # Validate frequency
    if args.freq not in ['D', 'W', 'M']:
        logging.error(f"Invalid frequency: {args.freq}. Must be 'D', 'W', or 'M'")
        sys.exit(1)
    
    # Print configuration
    print("\n" + "=" * 80)
    print("FORECASTER CONFIGURATION")
    print("=" * 80)
    print(f"• Frequency: {args.freq}")
    print(f"• Number of cutoffs: {args.n_cutoffs}")
    print(f"• Hyperparameter tuning: {'Enabled' if args.tune else 'Disabled'}")
    print(f"• Model: {args.model}")
    
    # Get frequency-specific parameters
    params = get_frequency_params(args.freq)
    
    # Start timing
    start_time = time.time()
    
    # Generate sample data
    df = generate_sample_data(freq=args.freq, periods=params['periods'])
    
    # Visualize the raw data
    visualize_data(df, args.freq, OUTPUTS_DIR)
    
    # Pre-create training_group column to ensure it exists
    df['training_group'] = df['product'].astype('category').cat.codes + 1
    logging.info(f"Pre-created training_group column with {df['training_group'].nunique()} unique values")
    
    # Configure the Runner
    runner = Runner()
    
    # Define basic parameters
    date_col = 'date'
    group_cols = ['product', 'store']
    signal_cols = ['sales']
    
    # Run the pipeline
    result_df = runner.run_pipeline(
        # Required parameters
        df=df,
        date_col=date_col,
        group_cols=group_cols,
        signal_cols=signal_cols,
        
        # Data preparation parameters
        target='sales',
        horizon=params['horizon'],
        freq=args.freq,
        n_cutoffs=args.n_cutoffs,
        complete_dataframe=True,
        smoothing=True,
        ma_window_size=params['ma_window_size'],
        fill_na=True,
        
        # Feature engineering parameters
        window_sizes=params['window_sizes'],
        lags=params['lags'],
        fill_lags=True,  # Fill lag values to avoid NaNs
        
        # Baseline parameters
        baseline_types=['MA'],  # Use only MA baseline for simplicity
        window_size=params['window_size'],
        
        # Forecaster parameters
        model=args.model,
        training_group='training_group',  # Use the pre-created training group column
        tune_hyperparameters=args.tune,
        use_feature_selection=True,
        n_best_features=10,
        use_guardrail=False,
        use_parallel=True
    )
    
    # Visualize the forecasts with cutoff indicators and train/test separation
    visualize_forecasts_by_cutoff(result_df, OUTPUTS_DIR, 'sales', 'prediction')
    
    # Save the results
    save_results(result_df, runner.metrics, args.freq, OUTPUTS_DIR)
    
    # Calculate execution time
    execution_time = time.time() - start_time
    
    # Print summary
    print("\n" + "=" * 80)
    print("FORECASTING RESULTS SUMMARY")
    print("=" * 80)
    print(f"\nExecution time: {execution_time:.2f} seconds")
    print(f"Data shape: {result_df.shape[0]} rows × {result_df.shape[1]} columns")
    
    if runner.metric_table is not None:
        print(f"\nMetrics Table:")
        print(runner.metric_table)
    
    print("\nAll results saved to:", OUTPUTS_DIR)
    print("=" * 80)


if __name__ == "__main__":
    main()
