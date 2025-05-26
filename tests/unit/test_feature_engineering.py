"""
Unit tests for the FeatureEngineering class focusing on run_feature_engineering function.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the FeatureEngineering class
from utils.feature_engineering import FeatureEngineering

class TestFeatureEngineering:
    """Test suite for FeatureEngineering class."""

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
                    
                    records.append({
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory,
                        'sample': 'train' if date < datetime(2022, 12, 1) else 'test',
                        'cutoff': datetime(2022, 12, 1)
                    })
        
        # Create DataFrame
        df = pd.DataFrame(records)
        
        # Ensure numeric columns are float64
        for col in ['sales', 'inventory']:
            df[col] = df[col].astype('float64')
            
        return df

    def test_run_feature_engineering_default_params(self, sample_data):
        """Test run_feature_engineering with default parameters."""
        # Initialize FeatureEngineering class
        fe = FeatureEngineering()
        
        # Define parameters for run_feature_engineering
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        freq = 'W-SAT'  # Weekly data on Saturdays
        
        # Run the function
        result_df = fe.run_feature_engineering(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that encoded feature columns exist
        assert 'feature_product' in result_df.columns, "Encoded product feature should exist"
        assert 'feature_store' in result_df.columns, "Encoded store feature should exist"
        
        # Check that date features exist
        assert 'feature_year' in result_df.columns, "Year feature should exist"
        assert 'feature_quarter' in result_df.columns, "Quarter feature should exist"
        assert 'feature_month' in result_df.columns, "Month feature should exist"
        assert 'feature_week' in result_df.columns, "Week feature should exist"
        
        # Check that period features exist
        assert 'feature_periods' in result_df.columns, "Periods feature should exist"
        assert 'feature_periods_expanding' in result_df.columns, "Periods expanding feature should exist"
        assert 'feature_periods_sqrt' in result_df.columns, "Periods sqrt feature should exist"
        
        # Check that moving average features exist
        for signal_col in ['sales', 'inventory']:
            for window_size in [4, 13]:  # Default window sizes
                assert f'{signal_col}_ma_{window_size}' in result_df.columns, f"MA feature for {signal_col} with window size {window_size} should exist"
        
        # Check that lag features exist
        for signal_col in ['sales', 'inventory']:
            for lag in [4, 13]:  # Default lags
                assert f'feature_{signal_col}_lag_{lag}' in result_df.columns, f"Lag feature for {signal_col} with lag {lag} should exist"
        
        # Check that coefficient of variance features exist
        assert f'feature_{target}_cov' in result_df.columns, "COV feature for sales should exist"
        
        # Check that quantile cluster features exist - based on actual column naming in the code
        assert f'feature_{target}_cluster' in result_df.columns, "Quantile cluster feature for sales should exist"
        
        # Check that history cluster features exist
        assert f'feature_{target}_history_cluster' in result_df.columns, "History cluster feature for sales should exist"
        
        # Check that train weights feature exists
        assert 'train_weight' in result_df.columns, "Train weights feature should exist"
        
        # Check that forecast lag number feature exists
        assert 'fcst_lag' in result_df.columns, "Forecast lag feature should exist"

    def test_run_feature_engineering_custom_params(self, sample_data):
        """Test run_feature_engineering with custom parameters."""
        # Initialize FeatureEngineering class
        fe = FeatureEngineering()
        
        # Define parameters for run_feature_engineering with custom values
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        freq = 'W-SAT'
        window_sizes = (3, 8)  # Custom window sizes
        lags = (2, 6)  # Custom lags
        fill_lags = True  # Enable lag filling
        n_clusters = 5  # Custom number of clusters
        
        # Run the function
        result_df = fe.run_feature_engineering(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq,
            window_sizes=window_sizes,
            lags=lags,
            fill_lags=fill_lags,
            n_clusters=n_clusters
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that window size features use custom values
        for signal_col in ['sales', 'inventory']:
            for window_size in window_sizes:
                assert f'{signal_col}_ma_{window_size}' in result_df.columns, f"MA feature for {signal_col} with window size {window_size} should exist"
                assert f'{signal_col}_min_{window_size}' in result_df.columns, f"Min feature for {signal_col} with window size {window_size} should exist"
                assert f'{signal_col}_max_{window_size}' in result_df.columns, f"Max feature for {signal_col} with window size {window_size} should exist"
        
        # Check that lag features use custom values
        for signal_col in ['sales', 'inventory']:
            for lag in lags:
                assert f'feature_{signal_col}_lag_{lag}' in result_df.columns, f"Lag feature for {signal_col} with lag {lag} should exist"
    
    def test_run_feature_engineering_multiple_group_cols(self, sample_data):
        """Test run_feature_engineering with multiple grouping columns."""
        # Initialize FeatureEngineering class
        fe = FeatureEngineering()
        
        # Add an extra grouping column to the sample data
        sample_data['category'] = np.where(sample_data['product'].isin(['P001', 'P002']), 'Category A', 'Category B')
        
        # Define parameters with multiple group columns
        group_cols = ['product', 'store', 'category']
        date_col = 'date'
        target = 'sales'
        freq = 'W-SAT'
        
        # Run the function
        result_df = fe.run_feature_engineering(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Verify that features are created correctly when using multiple group columns
        assert 'feature_category' in result_df.columns, "Encoded category feature should exist"
        
        # Check that features based on grouping use all group columns
        # We can verify this by checking the number of unique values in a feature that depends on grouping
        # The feature_periods column should have one value per group
        group_count = sample_data[group_cols].drop_duplicates().shape[0]
        periods_count = result_df.groupby(group_cols)['feature_periods'].nunique().sum()
        assert periods_count >= group_count, "feature_periods should be calculated for each group"
        
    def test_run_feature_engineering_with_missing_values(self):
        """Test run_feature_engineering with missing values in the dataset."""
        # Create a dataset with missing values
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i*7) for i in range(20)]  # Weekly data
        
        # Create records with some missing values
        records = []
        for i, date in enumerate(dates):
            # Make every 5th sales value NaN and ensure others are float
            sales = float(np.random.randint(10, 100)) if i % 5 != 0 else np.nan
            # Make every 7th inventory value NaN and ensure others are float
            inventory = float(np.random.randint(50, 200)) if i % 7 != 0 else np.nan
            
            records.append({
                'date': date,
                'product': 'P001',
                'store': 'S01',
                'sales': sales,
                'inventory': inventory,
                'sample': 'train' if date < datetime(2022, 5, 1) else 'test',
                'cutoff': datetime(2022, 5, 1)
            })
        
        # Create DataFrame with missing values
        df_with_missing = pd.DataFrame(records)
        
        # Ensure numeric columns are float64
        for col in ['sales', 'inventory']:
            df_with_missing[col] = df_with_missing[col].astype('float64')
        
        # Initialize FeatureEngineering class
        fe = FeatureEngineering()
        
        # Define parameters for run_feature_engineering
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        freq = 'W-SAT'
        
        # Run the function
        result_df = fe.run_feature_engineering(
            df=df_with_missing,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that features are created despite missing values
        assert 'feature_periods' in result_df.columns, "Periods feature should exist"
        assert 'feature_year' in result_df.columns, "Year feature should exist"
        
        # Check that moving average features handle missing values appropriately
        assert 'sales_ma_4' in result_df.columns, "MA feature for sales should exist"
        # Verify that moving averages help fill some missing values
        assert result_df['sales_ma_4'].notna().sum() > df_with_missing['sales'].notna().sum(), "MA should help fill some missing values"
        
    def test_run_feature_engineering_different_freq(self, sample_data):
        """Test run_feature_engineering with different frequency specifications."""
        # Initialize FeatureEngineering class
        fe = FeatureEngineering()
        
        # Define parameters for monthly frequency
        group_cols = ['product', 'store']
        date_col = 'date'
        target = 'sales'
        freq = 'M'  # Monthly frequency
        
        # Run the function with monthly frequency
        result_df = fe.run_feature_engineering(
            df=sample_data,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq
        )
        
        # Assertions
        assert isinstance(result_df, pd.DataFrame), "Result should be a DataFrame"
        
        # Check that date features exist and are appropriate for monthly data
        assert 'feature_year' in result_df.columns, "Year feature should exist"
        assert 'feature_quarter' in result_df.columns, "Quarter feature should exist"
        assert 'feature_month' in result_df.columns, "Month feature should exist"
        # Weekly feature should not exist for monthly data
        assert 'feature_week' not in result_df.columns, "Week feature should not exist for monthly data"
        
        # Check that other features still exist
        assert 'feature_periods' in result_df.columns, "Periods feature should exist"
        assert 'fcst_lag' in result_df.columns, "Forecast lag feature should exist"

if __name__ == "__main__":
    pytest.main(["-v", "test_feature_engineering.py"])
