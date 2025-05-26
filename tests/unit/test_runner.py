"""
Unit tests for the Runner class focusing on pipeline orchestration with different data frequencies.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the Runner class
from utils.runner import Runner

class TestRunner:
    """Test suite for Runner class."""

    @pytest.fixture
    def daily_data(self):
        """Create a sample dataset with daily frequency for testing the Runner class."""
        # Create date range - daily data for 60 days
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(60)]
        
        # Create product IDs and store IDs (keeping dataset small for test performance)
        product_ids = ['P001']
        store_ids = ['S01']
        
        # Create all combinations of dates, products, and stores
        records = []
        for date in dates:
            for product_id in product_ids:
                for store_id in store_ids:
                    # Generate some sample data with a bit of pattern
                    # Make sales follow a seasonal pattern with some noise
                    day_of_year = date.timetuple().tm_yday
                    seasonal_component = np.sin(day_of_year / 365 * 2 * np.pi) * 20 + 50
                    
                    # Add day of week seasonality for daily data
                    weekday_factor = 1.0 + 0.2 * (date.weekday() < 5)  # Higher on weekdays
                    
                    sales = float(seasonal_component * weekday_factor + np.random.normal(0, 3))
                    inventory = float(sales * 2 + np.random.normal(0, 5))
                    
                    # Create a record
                    record = {
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory
                    }
                    records.append(record)
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    @pytest.fixture
    def weekly_data(self):
        """Create a sample dataset with weekly frequency for testing the Runner class."""
        # Create date range - weekly data for 52 weeks
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i*7) for i in range(52)]
        
        # Create product IDs and store IDs (keeping dataset small for test performance)
        product_ids = ['P001']
        store_ids = ['S01']
        
        # Create all combinations of dates, products, and stores
        records = []
        for date in dates:
            for product_id in product_ids:
                for store_id in store_ids:
                    # Generate some sample data with a bit of pattern
                    # Make sales follow a seasonal pattern with some noise
                    week_of_year = date.isocalendar()[1]
                    seasonal_component = np.sin(week_of_year / 52 * 2 * np.pi) * 20 + 50
                    
                    sales = float(seasonal_component + np.random.normal(0, 5))
                    inventory = float(sales * 2 + np.random.normal(0, 10))
                    
                    # Create a record
                    record = {
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory
                    }
                    records.append(record)
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    @pytest.fixture
    def monthly_data(self):
        """Create a sample dataset with monthly frequency for testing the Runner class."""
        # Create date range - monthly data for 24 months
        start_date = datetime(2022, 1, 1)
        dates = []
        for i in range(24):
            year = start_date.year + ((start_date.month - 1 + i) // 12)
            month = ((start_date.month - 1 + i) % 12) + 1
            dates.append(datetime(year, month, 1))
        
        # Create product IDs and store IDs (keeping dataset small for test performance)
        product_ids = ['P001']
        store_ids = ['S01']
        
        # Create all combinations of dates, products, and stores
        records = []
        for date in dates:
            for product_id in product_ids:
                for store_id in store_ids:
                    # Generate some sample data with a bit of pattern
                    # Make sales follow a seasonal pattern with some noise
                    month = date.month
                    seasonal_component = np.sin(month / 12 * 2 * np.pi) * 30 + 100
                    
                    sales = float(seasonal_component + np.random.normal(0, 10))
                    inventory = float(sales * 2 + np.random.normal(0, 20))
                    
                    # Create a record
                    record = {
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory
                    }
                    records.append(record)
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    def test_init(self):
        """Test initialization of Runner class."""
        runner = Runner()
        
        # Check that metrics and final_df are initialized to None
        assert runner.metrics is None
        assert runner.metric_table is None
        assert runner.final_df is None

    def test_run_pipeline_daily_data(self, daily_data):
        """Test run_pipeline method with daily data."""
        runner = Runner()
        
        # Define basic parameters
        date_col = 'date'
        group_cols = ['product', 'store']
        signal_cols = ['sales']
        
        # Run with parameters optimized for daily data
        result_df = runner.run_pipeline(
            # Required parameters
            df=daily_data,
            date_col=date_col,
            group_cols=group_cols,
            signal_cols=signal_cols,
            
            # Data preparation parameters
            target='sales',  # Specify target explicitly
            horizon=7,       # 7-day forecast horizon for daily data
            freq='D',        # Use daily frequency
            n_cutoffs=1,     # Use single cutoff for testing
            complete_dataframe=True,
            smoothing=True,
            ma_window_size=7, # Use 7-day smoothing window for daily data
            fill_na=True,
            
            # Feature engineering parameters
            window_sizes=(3, 7),  # Appropriate window sizes for daily data
            lags=(1, 7),         # 1-7 day lags for daily data
            fill_lags=False,
            
            # Baseline parameters
            baseline_types=['MA'],  # Keep simple for testing
            window_size=7,          # 7-day window for moving average
            
            # Forecaster parameters
            model='LGBM',
            tune_hyperparameters=False,  # Disable for faster tests
            use_feature_selection=True,
            n_best_features=10,
            use_guardrail=False,
            use_parallel=False  # Disable for testing
        )
        
        # Check that the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that required columns exist
        assert 'date' in result_df.columns
        assert 'product' in result_df.columns
        assert 'store' in result_df.columns
        assert 'sales' in result_df.columns
        
        # Check that metrics were generated
        assert runner.metrics is not None
        assert runner.metric_table is not None
        
        # Check that metrics table is a DataFrame
        assert isinstance(runner.metric_table, pd.DataFrame)

    def test_run_pipeline_weekly_data(self, weekly_data):
        """Test run_pipeline method with weekly data."""
        runner = Runner()
        
        # Define basic parameters
        date_col = 'date'
        group_cols = ['product', 'store']
        signal_cols = ['sales']
        
        # Run with parameters optimized for weekly data
        result_df = runner.run_pipeline(
            # Required parameters
            df=weekly_data,
            date_col=date_col,
            group_cols=group_cols,
            signal_cols=signal_cols,
            
            # Data preparation parameters
            target='sales',  # Specify target explicitly
            horizon=4,       # 4-week forecast horizon for weekly data
            freq='W',        # Use weekly frequency
            n_cutoffs=1,     # Use single cutoff for testing
            complete_dataframe=True,
            smoothing=True,
            ma_window_size=4, # Use 4-week smoothing window for weekly data
            fill_na=True,
            
            # Feature engineering parameters
            window_sizes=(2, 8),   # Appropriate window sizes for weekly data
            lags=(1, 8),          # 1-8 week lags for weekly data
            fill_lags=False,
            
            # Baseline parameters
            baseline_types=['MA'],  # Keep simple for testing
            window_size=4,          # 4-week window for moving average
            
            # Forecaster parameters
            model='LGBM',
            tune_hyperparameters=False,  # Disable for faster tests
            use_feature_selection=True,
            n_best_features=10,
            use_guardrail=False,
            use_parallel=False  # Disable for testing
        )
        
        # Check that the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that required columns exist
        assert 'date' in result_df.columns
        assert 'sales' in result_df.columns
        
        # Check that metrics were generated
        assert runner.metrics is not None
        assert runner.metric_table is not None

    def test_run_pipeline_monthly_data(self, monthly_data):
        """Test run_pipeline method with monthly data."""
        runner = Runner()
        
        # Define basic parameters
        date_col = 'date'
        group_cols = ['product', 'store']
        signal_cols = ['sales']
        
        # Run with parameters optimized for monthly data
        result_df = runner.run_pipeline(
            # Required parameters
            df=monthly_data,
            date_col=date_col,
            group_cols=group_cols,
            signal_cols=signal_cols,
            
            # Data preparation parameters
            target='sales',  # Specify target explicitly
            horizon=3,       # 3-month forecast horizon for monthly data
            freq='M',        # Use monthly frequency
            n_cutoffs=1,     # Use single cutoff for testing
            complete_dataframe=True,
            smoothing=True,
            ma_window_size=3, # Use 3-month smoothing window for monthly data
            fill_na=True,
            
            # Feature engineering parameters
            window_sizes=(2, 12),  # Appropriate window sizes for monthly data
            lags=(1, 12),         # 1-12 month lags for monthly data
            fill_lags=False,
            
            # Baseline parameters
            baseline_types=['MA'],  # Keep simple for testing
            window_size=3,          # 3-month window for moving average
            
            # Forecaster parameters
            model='LGBM',
            tune_hyperparameters=False,  # Disable for faster tests
            use_feature_selection=True,
            n_best_features=10,
            use_guardrail=False,
            use_parallel=False  # Disable for testing
        )
        
        # Check that the result is a DataFrame
        assert isinstance(result_df, pd.DataFrame)
        
        # Check that required columns exist
        assert 'date' in result_df.columns
        assert 'sales' in result_df.columns
        
        # Check that metrics were generated
        assert runner.metrics is not None
        assert runner.metric_table is not None

if __name__ == "__main__":
    pytest.main(["-v", "test_runner.py"])
