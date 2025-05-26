"""
Unit tests for the DataPreparation class focusing on run_data_preparation function.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the DataPreparation class
from utils.data_preparation import DataPreparation

class TestDataPreparation:
    """Test suite for DataPreparation class."""

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
            for product in product_ids:
                for store in store_ids:
                    # Generate some random sales data
                    sales = np.random.randint(10, 100)
                    inventory = np.random.randint(50, 200)
                    
                    records.append({
                        'date': date,
                        'product': product,
                        'store': store,
                        'sales': sales,
                        'inventory': inventory
                    })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        return df

    def test_run_data_preparation_default_params(self, sample_data):
        """Test run_data_preparation with default parameters."""
        # Initialize DataPreparation class
        dp = DataPreparation()
        
        # Define parameters for run_data_preparation
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        horizon = 4  # 4-week horizon
        
        # Run the function
        result_df = dp.run_data_preparation(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            horizon=horizon
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        assert 'cutoff' in result_df.columns, "Result should have a 'cutoff' column"
        assert 'sample' in result_df.columns, "Result should have a 'sample' column"
        
        # Check that all groups have at least horizon+1 test samples for the last cutoff
        last_cutoff = result_df['cutoff'].max()
        for _, group_df in result_df[result_df['cutoff'] == last_cutoff].groupby(group_cols):
            test_samples = group_df[group_df['sample'] == 'test']
            assert len(test_samples) >= horizon, f"Each group should have at least {horizon} test samples"

    def test_run_data_preparation_custom_freq(self, sample_data):
        """Test run_data_preparation with custom frequency."""
        # Initialize DataPreparation class
        dp = DataPreparation()
        
        # Define parameters for run_data_preparation
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        horizon = 4
        freq = 'W-MON'  # Weekly on Monday
        
        # Run the function
        result_df = dp.run_data_preparation(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            horizon=horizon,
            freq=freq
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check if date differences align with the specified frequency
        last_cutoff = result_df['cutoff'].max()
        for _, group_df in result_df[result_df['cutoff'] == last_cutoff].groupby(group_cols):
            test_samples = group_df[group_df['sample'] == 'test'].sort_values(by=date_col)
            if len(test_samples) >= 2:
                date_diffs = test_samples[date_col].diff().dropna()
                # For weekly frequency, the differences should be close to 7 days
                assert all(diff.days in [6, 7, 8] for diff in date_diffs), "Date differences should match weekly frequency"

    def test_run_data_preparation_no_smoothing(self, sample_data):
        """Test run_data_preparation with smoothing disabled."""
        # Initialize DataPreparation class
        dp = DataPreparation()
        
        # Define parameters for run_data_preparation
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        horizon = 4
        smoothing = False
        
        # Run the function
        result_df = dp.run_data_preparation(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            horizon=horizon,
            smoothing=smoothing
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that filled columns are not present (since smoothing is disabled)
        filled_cols = [col for col in result_df.columns if col.startswith('filled_')]
        assert len(filled_cols) == 0, "There should be no 'filled_' columns when smoothing is disabled"

    def test_run_data_preparation_fewer_cutoffs(self, sample_data):
        """Test run_data_preparation with fewer cutoffs."""
        # Initialize DataPreparation class
        dp = DataPreparation()
        
        # Define parameters for run_data_preparation
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        horizon = 4
        n_cutoffs = 6  # Reduced number of cutoffs
        
        # Run the function
        result_df = dp.run_data_preparation(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            horizon=horizon,
            n_cutoffs=n_cutoffs
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that the number of unique cutoffs is less than or equal to n_cutoffs
        unique_cutoffs = result_df['cutoff'].unique()
        assert len(unique_cutoffs) <= n_cutoffs, f"There should be at most {n_cutoffs} cutoffs"

    def test_run_data_preparation_missing_values(self):
        """Test run_data_preparation with missing values in the dataset."""
        # Create a dataset with missing values
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i*7) for i in range(20)]  # Weekly data
        
        # Create records with some missing values
        records = []
        for i, date in enumerate(dates):
            sales = np.random.randint(10, 100) if i % 5 != 0 else np.nan  # Make every 5th sales value NaN
            inventory = np.random.randint(50, 200) if i % 7 != 0 else np.nan  # Make every 7th inventory value NaN
            
            records.append({
                'date': date,
                'product': 'P001',
                'store': 'S01',
                'sales': sales,
                'inventory': inventory
            })
        
        # Create DataFrame with missing values
        df_with_missing = pd.DataFrame(records)
        
        # Initialize DataPreparation class
        dp = DataPreparation()
        
        # Define parameters for run_data_preparation
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        horizon = 4
        
        # Run the function
        result_df = dp.run_data_preparation(
            df=df_with_missing,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            horizon=horizon
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that filled columns exist and have fewer NaN values than the original
        filled_sales_col = 'filled_sales'
        assert filled_sales_col in result_df.columns, f"'{filled_sales_col}' column should exist"
        
        # Get count of NaN values in original data
        original_na_count = df_with_missing['sales'].isna().sum()
        
        # Calculate the percentage of NaN values in the filled column for non-horizon rows only
        # Horizon rows are those where 'sample' = 'test'
        train_rows = result_df[result_df['sample'] == 'train']
        train_filled_na_count = train_rows[filled_sales_col].isna().sum()
        
        # Assert that we have some improvement in the training data portion
        print(f"Original NaN count: {original_na_count}, Filled NaN count in train rows: {train_filled_na_count}")
        assert train_filled_na_count <= original_na_count, f"'{filled_sales_col}' should have fewer or equal NaN values in training data compared to original data"
        
        # Check that we have some valid values in the filled column overall
        assert result_df[filled_sales_col].notna().sum() > 0, f"'{filled_sales_col}' should have some valid values"
        
        # It's expected that test/horizon rows may contain NaNs
        test_rows = result_df[result_df['sample'] == 'test']
        print(f"Number of horizon/test rows: {len(test_rows)}, NaN count in test rows: {test_rows[filled_sales_col].isna().sum()}")


if __name__ == "__main__":
    pytest.main(["-v", "test_data_preparation.py"])
