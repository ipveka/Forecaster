#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script to demonstrate horizon filtering functionality.
This script shows how the new filter_by_horizon_param parameter works.
"""

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.runner import Runner
from utils.forecaster_utils import generate_sample_data

def test_horizon_filtering():
    """Test the horizon filtering functionality."""
    print("=" * 80)
    print("TESTING HORIZON FILTERING FUNCTIONALITY")
    print("=" * 80)
    
    # Generate sample data
    print("\n📊 Generating sample data...")
    df = generate_sample_data(freq="W", periods=52)  # 1 year of weekly data
    print(f"   ✓ Generated {len(df)} rows of sample data")
    
    # Initialize runner
    runner = Runner(verbose=True)
    
    # Define parameters
    date_col = "date"
    group_cols = ["product", "store"]
    signal_cols = ["sales"]
    target = "sales"
    horizon = 12  # 12-week forecast horizon
    
    print(f"\n🎯 Configuration:")
    print(f"   • Horizon: {horizon} periods")
    print(f"   • Horizon column will be created in data preparation")
    print(f"   • Will filter to horizon <= {horizon} (using existing horizon parameter)")
    
    # Run pipeline with horizon filtering
    result_df = runner.run_pipeline(
        # Required parameters
        df=df,
        date_col=date_col,
        group_cols=group_cols,
        signal_cols=signal_cols,
        # Data preparation parameters
        target=target,
        horizon=horizon,  # This will be used for filtering
        freq="W",
        n_cutoffs=2,
        complete_dataframe=True,
        smoothing=False,
        dp_window_size=13,
        # Feature engineering parameters
        fe_window_size=(4, 13),
        lags=(4, 13),
        fill_lags=True,
        n_clusters=10,
        # Baseline parameters
        baseline_types=["MA"],
        bs_window_size=13,
        # Forecaster parameters
        model="LGBM",
        training_group="training_group",
        tune_hyperparameters=False,
        use_feature_selection=False,
        use_lags=True,
        n_best_features=20,
        use_guardrail=False,
        use_parallel=False,
    )
    
    # Analyze results
    print(f"\n📈 Results Analysis:")
    print(f"   • Final dataset shape: {result_df.shape}")
    
    # Check horizon distribution in test data
    test_data = result_df[result_df["sample"] == "test"]
    if len(test_data) > 0 and "horizon" in test_data.columns:
        horizon_dist = test_data["horizon"].value_counts().sort_index()
        print(f"   • Horizon distribution in test data:")
        for h, count in horizon_dist.items():
            print(f"     - Horizon {h}: {count:,} rows")
        
        max_horizon = test_data["horizon"].max()
        print(f"   • Maximum horizon in final dataset: {max_horizon}")
        print(f"   • Expected maximum horizon: {horizon} (due to filtering)")
        
        if max_horizon <= horizon:
            print(f"   ✅ Horizon filtering worked correctly!")
        else:
            print(f"   ❌ Horizon filtering may not have worked as expected")
    else:
        print(f"   ⚠️  No test data or horizon column found")
    
    print(f"\n" + "=" * 80)
    print("HORIZON FILTERING TEST COMPLETED")
    print("=" * 80)

if __name__ == "__main__":
    test_horizon_filtering()
