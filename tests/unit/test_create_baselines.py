"""
Unit tests for the CreateBaselines class focusing on baseline creation functions.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the CreateBaselines class
from utils.create_baselines import CreateBaselines

class TestCreateBaselines:
    """Test suite for CreateBaselines class."""

    @pytest.fixture
    def sample_data(self):
        """Create a sample dataset for testing."""
        # Create date range
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i*7) for i in range(52)]  # Weekly data for one year
        
        # Create product IDs
        product_ids = ['P001', 'P002', 'P003']
        
        # Create store IDs
        store_ids = ['S01', 'S02']
        
        # Create all combinations of dates, products, and stores
        records = []
        for date in dates:
            for product_id in product_ids:
                for store_id in store_ids:
                    # Generate some sample data
                    sales = float(np.random.randint(10, 100))  # Convert to float
                    inventory = float(np.random.randint(50, 200))  # Convert to float
                    
                    # Create some basic features for regression models
                    feature_day = float(date.day)
                    feature_month = float(date.month)
                    feature_product = 1.0 if product_id == 'P001' else (2.0 if product_id == 'P002' else 3.0)
                    feature_store = 1.0 if store_id == 'S01' else 2.0
                    
                    records.append({
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory,
                        'feature_day': feature_day,
                        'feature_month': feature_month,
                        'feature_product': feature_product,
                        'feature_store': feature_store,
                        'sample': 'train' if date < datetime(2022, 12, 1) else 'test',
                        'cutoff': datetime(2022, 12, 1)
                    })
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    def test_create_ma_baseline(self, sample_data):
        """Test create_ma_baseline function."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for create_ma_baseline
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        window_size = 4
        
        # Run the function
        result_df = cb.create_ma_baseline(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=signal_cols,
            window_size=window_size
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that baseline columns exist for each signal column
        for signal_col in signal_cols:
            assert f'baseline_{signal_col}' in result_df.columns, f"Baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}' in result_df.columns, f"Feature baseline column for {signal_col} should exist"
        
        # Check that values are filled correctly for test data
        test_data = result_df[result_df['sample'] == 'test']
        for signal_col in signal_cols:
            assert not test_data[f'baseline_{signal_col}'].isna().any(), f"Baseline for {signal_col} should not have NaN values in test data"
            assert not test_data[f'feature_baseline_{signal_col}'].isna().any(), f"Feature baseline for {signal_col} should not have NaN values in test data"
            # Check that baseline and feature baseline are equal for test data
            assert (test_data[f'baseline_{signal_col}'] == test_data[f'feature_baseline_{signal_col}']).all(), \
                f"Baseline and feature baseline should be equal for {signal_col} in test data"

    def test_create_lr_baseline(self, sample_data):
        """Test create_lr_baseline function."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for create_lr_baseline
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        feature_cols = ['feature_day', 'feature_month', 'feature_product', 'feature_store']
        
        # Run the function
        result_df = cb.create_lr_baseline(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=signal_cols,
            feature_cols=feature_cols,
            debug=False
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that baseline columns exist for each signal column
        for signal_col in signal_cols:
            assert f'baseline_{signal_col}_lr' in result_df.columns, f"LR baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}_lr' in result_df.columns, f"LR feature baseline column for {signal_col} should exist"
        
        # Check that values are filled correctly for test data
        test_data = result_df[result_df['sample'] == 'test']
        for signal_col in signal_cols:
            assert not test_data[f'baseline_{signal_col}_lr'].isna().any(), f"LR baseline for {signal_col} should not have NaN values in test data"
            assert not test_data[f'feature_baseline_{signal_col}_lr'].isna().any(), f"LR feature baseline for {signal_col} should not have NaN values in test data"
            # Check that baseline and feature baseline are equal for test data
            assert (test_data[f'baseline_{signal_col}_lr'] == test_data[f'feature_baseline_{signal_col}_lr']).all(), \
                f"LR baseline and feature baseline should be equal for {signal_col} in test data"
        
        # Check that train data contains the original signal values in the baseline columns
        train_data = result_df[result_df['sample'] == 'train']
        for signal_col in signal_cols:
            assert (train_data[f'baseline_{signal_col}_lr'] == train_data[signal_col]).all(), \
                f"LR baseline should equal original {signal_col} in train data"
            assert (train_data[f'feature_baseline_{signal_col}_lr'] == train_data[signal_col]).all(), \
                f"LR feature baseline should equal original {signal_col} in train data"

    def test_create_lgbm_baseline(self, sample_data):
        """Test create_lgbm_baseline function."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for create_lgbm_baseline
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        feature_cols = ['feature_day', 'feature_month', 'feature_product', 'feature_store']
        
        # Run the function
        result_df = cb.create_lgbm_baseline(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=signal_cols,
            feature_cols=feature_cols,
            debug=False
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that baseline columns exist for each signal column
        for signal_col in signal_cols:
            assert f'baseline_{signal_col}_lgbm' in result_df.columns, f"LGBM baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}_lgbm' in result_df.columns, f"LGBM feature baseline column for {signal_col} should exist"
        
        # Check that values are filled correctly for test data
        test_data = result_df[result_df['sample'] == 'test']
        for signal_col in signal_cols:
            assert not test_data[f'baseline_{signal_col}_lgbm'].isna().any(), f"LGBM baseline for {signal_col} should not have NaN values in test data"
            assert not test_data[f'feature_baseline_{signal_col}_lgbm'].isna().any(), f"LGBM feature baseline for {signal_col} should not have NaN values in test data"
            # Check that baseline and feature baseline are equal for test data
            assert (test_data[f'baseline_{signal_col}_lgbm'] == test_data[f'feature_baseline_{signal_col}_lgbm']).all(), \
                f"LGBM baseline and feature baseline should be equal for {signal_col} in test data"
        
        # Check that train data contains the original signal values in the baseline columns
        train_data = result_df[result_df['sample'] == 'train']
        for signal_col in signal_cols:
            assert (train_data[f'baseline_{signal_col}_lgbm'] == train_data[signal_col]).all(), \
                f"LGBM baseline should equal original {signal_col} in train data"
            assert (train_data[f'feature_baseline_{signal_col}_lgbm'] == train_data[signal_col]).all(), \
                f"LGBM feature baseline should equal original {signal_col} in train data"

    def test_run_baselines_ma_only(self, sample_data):
        """Test run_baselines function with MA baseline only."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for run_baselines
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        baseline_types = ['MA']
        window_size = 4
        
        # Run the function
        result_df = cb.run_baselines(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=signal_cols,
            baseline_types=baseline_types,
            window_size=window_size
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that MA baseline columns exist
        for signal_col in signal_cols:
            assert f'baseline_{signal_col}' in result_df.columns, f"MA baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}' in result_df.columns, f"MA feature baseline column for {signal_col} should exist"
        
        # Check that LR and LGBM baseline columns don't exist
        for signal_col in signal_cols:
            assert f'baseline_{signal_col}_lr' not in result_df.columns, f"LR baseline column should not exist"
            assert f'baseline_{signal_col}_lgbm' not in result_df.columns, f"LGBM baseline column should not exist"

    def test_run_baselines_all_types(self, sample_data):
        """Test run_baselines function with all baseline types."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for run_baselines
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        baseline_types = ['MA', 'LR', 'ML']
        window_size = 4
        feature_cols = ['feature_day', 'feature_month', 'feature_product', 'feature_store']
        
        # Run the function
        result_df = cb.run_baselines(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=signal_cols,
            baseline_types=baseline_types,
            window_size=window_size,
            feature_cols=feature_cols
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that all baseline columns exist
        for signal_col in signal_cols:
            # MA baselines
            assert f'baseline_{signal_col}' in result_df.columns, f"MA baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}' in result_df.columns, f"MA feature baseline column for {signal_col} should exist"
            
            # LR baselines
            assert f'baseline_{signal_col}_lr' in result_df.columns, f"LR baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}_lr' in result_df.columns, f"LR feature baseline column for {signal_col} should exist"
            
            # LGBM baselines
            assert f'baseline_{signal_col}_lgbm' in result_df.columns, f"LGBM baseline column for {signal_col} should exist"
            assert f'feature_baseline_{signal_col}_lgbm' in result_df.columns, f"LGBM feature baseline column for {signal_col} should exist"

    def test_run_baselines_feature_cols_validation(self, sample_data):
        """Test that run_baselines raises an error when feature_cols is not provided for LR or ML baselines."""
        # Initialize CreateBaselines class
        cb = CreateBaselines()
        
        # Define parameters for run_baselines
        group_cols = ['product', 'store']
        date_col = 'date'
        signal_cols = ['sales', 'inventory']
        baseline_types = ['LR']  # Only LR baseline
        
        # Run the function with missing feature_cols
        with pytest.raises(ValueError):
            cb.run_baselines(
                df=sample_data,
                group_cols=group_cols,
                date_col=date_col,
                signal_cols=signal_cols,
                baseline_types=baseline_types
            )

if __name__ == "__main__":
    pytest.main(["-v", "test_create_baselines.py"])
