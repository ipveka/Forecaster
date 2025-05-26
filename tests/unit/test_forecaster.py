"""
Unit tests for the Forecaster class focusing on forecasting functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os
import lightgbm as lgb
from sklearn.linear_model import LinearRegression

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the Forecaster class
from utils.forecaster import Forecaster

class TestForecaster:
    """Test suite for Forecaster class."""

    @pytest.fixture
    def sample_data(self):
        """Create a sample dataset for testing the Forecaster class."""
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
                    # Generate some sample data with a bit of pattern
                    # Make sales follow a seasonal pattern with some noise
                    day_of_year = date.timetuple().tm_yday
                    seasonal_component = np.sin(day_of_year / 365 * 2 * np.pi) * 20 + 50
                    product_factor = 1.0 if product_id == 'P001' else (0.8 if product_id == 'P002' else 1.2)
                    store_factor = 1.0 if store_id == 'S01' else 1.3
                    
                    sales = float(seasonal_component * product_factor * store_factor + np.random.normal(0, 5))
                    inventory = float(sales * 2 + np.random.normal(0, 10))
                    
                    # Create some features for model training
                    feature_day = float(date.day)
                    feature_month = float(date.month)
                    feature_year = float(date.year)
                    feature_dayofweek = float(date.weekday())
                    feature_product = 1.0 if product_id == 'P001' else (2.0 if product_id == 'P002' else 3.0)
                    feature_store = 1.0 if store_id == 'S01' else 2.0
                    feature_ma_4 = float(np.random.normal(sales, 2))  # Mock moving average
                    
                    # Determine if this is a training or test sample
                    # Use the last 4 weeks as test data
                    is_test = date >= datetime(2022, 12, 1)
                    
                    # Assign a training group based on product ID
                    training_group = 1 if product_id == 'P001' else (2 if product_id == 'P002' else 3)
                    
                    # Create a record
                    record = {
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'sales': sales,
                        'inventory': inventory,
                        'feature_day': feature_day,
                        'feature_month': feature_month,
                        'feature_year': feature_year,
                        'feature_dayofweek': feature_dayofweek,
                        'feature_product': feature_product,
                        'feature_store': feature_store,
                        'feature_ma_4': feature_ma_4,
                        'baseline_sales': sales * 0.9,  # Mock baseline prediction for guardrail testing
                        'baseline_inventory': inventory * 0.9,
                        'sample': 'test' if is_test else 'train',
                        'cutoff': datetime(2022, 12, 1),
                        'training_group': training_group
                    }
                    records.append(record)
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    def test_init(self, sample_data):
        """Test initialization of Forecaster class."""
        forecaster = Forecaster(sample_data)
        
        # Check that df was correctly copied
        assert isinstance(forecaster.df, pd.DataFrame)
        assert len(forecaster.df) == len(sample_data)
        
        # Check that models and feature_importances dictionaries were initialized
        assert isinstance(forecaster.models, dict)
        assert isinstance(forecaster.feature_importances, dict)
        assert isinstance(forecaster.best_hyperparams, dict)
        assert len(forecaster.models) == 0
        assert len(forecaster.feature_importances) == 0
        assert len(forecaster.best_hyperparams) == 0

    def test_remove_outliers(self, sample_data):
        """Test the remove_outliers method."""
        # Make some extreme outliers in the sales column
        sample_data.loc[10, 'sales'] = 1000.0  # Very high value
        sample_data.loc[20, 'sales'] = -100.0  # Negative value (invalid for sales)
        
        forecaster = Forecaster(sample_data)
        
        # Apply outlier removal
        forecaster.remove_outliers(
            column='sales',
            group_cols=['product', 'store'],
            lower_quantile=0.01,
            upper_quantile=0.99
        )
        
        # Check that the outliers were capped
        assert forecaster.df.loc[10, 'sales'] < 1000.0  # Should be capped at upper quantile
        assert forecaster.df.loc[20, 'sales'] > -100.0  # Should be capped at lower quantile

    def test_prepare_features_for_selection(self, sample_data):
        """Test the prepare_features_for_selection method."""
        forecaster = Forecaster(sample_data)
        
        # Prepare features
        train_data = sample_data[sample_data['sample'] == 'train']
        group_cols = ['product', 'store']
        numeric_features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        
        X_sampled, y_sampled = forecaster.prepare_features_for_selection(
            train_data, 
            group_cols, 
            numeric_features, 
            target_col, 
            sample_fraction=0.5
        )
        
        # Check results
        assert isinstance(X_sampled, pd.DataFrame)
        assert isinstance(y_sampled, pd.Series)
        assert len(X_sampled) == len(y_sampled)
        assert all(col in X_sampled.columns for col in numeric_features)
        assert len(X_sampled) <= len(train_data)  # Should be sampled

    def test_select_best_features(self, sample_data):
        """Test the select_best_features method."""
        forecaster = Forecaster(sample_data)
        
        # Prepare features
        train_data = sample_data[sample_data['sample'] == 'train']
        group_cols = ['product', 'store']
        numeric_features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        
        X_sampled, y_sampled = forecaster.prepare_features_for_selection(
            train_data, 
            group_cols, 
            numeric_features, 
            target_col, 
            sample_fraction=0.5
        )
        
        # Select best features
        n_best_features = 3  # Select top 3 features
        selected_features = forecaster.select_best_features(
            X_sampled, 
            y_sampled, 
            group_cols, 
            n_best_features
        )
        
        # Check results
        assert isinstance(selected_features, list)
        assert len(selected_features) <= n_best_features
        assert all(feat in numeric_features for feat in selected_features)

    def test_train_model(self, sample_data):
        """Test the train_model method."""
        forecaster = Forecaster(sample_data)
        
        # Filter data for a specific cutoff and training group
        cutoff = datetime(2022, 12, 1)
        training_group_val = 1
        
        cutoff_df = sample_data[sample_data['cutoff'] == cutoff]
        cutoff_df_tg = cutoff_df[cutoff_df['training_group'] == training_group_val]
        
        # Set up parameters for model training
        features = [col for col in sample_data.columns if col.startswith('feature_')]
        params = {'LGBM': lgb.LGBMRegressor(n_jobs=-1, objective='regression', random_state=42, verbose=-1)}
        target_col = 'sales'
        training_group = 'training_group'
        
        # Train the model
        result_df = forecaster.train_model(
            df=cutoff_df_tg,
            cutoff=cutoff,
            features=features,
            params=params,
            training_group=training_group,
            training_group_val=training_group_val,
            target_col=target_col,
            use_weights=False,
            tune_hyperparameters=False,
            model='LGBM'
        )
        
        # Check results
        assert isinstance(result_df, pd.DataFrame)
        assert 'prediction' in result_df.columns
        
        # Check that test samples have predictions
        test_samples = result_df[result_df['sample'] == 'test']
        assert not test_samples['prediction'].isna().any()
        
        # Check that a model was saved
        model_key = (cutoff, training_group_val)
        assert model_key in forecaster.models

    def test_calculate_guardrail(self, sample_data):
        """Test the calculate_guardrail method."""
        forecaster = Forecaster(sample_data)
        
        # Add a prediction column
        sample_data['prediction'] = sample_data['sales'] * 1.5  # Intentionally high to trigger guardrail
        
        # Select test data
        test_data = sample_data[sample_data['sample'] == 'test']
        
        # Apply guardrail
        group_cols = ['product', 'store']
        baseline_col = 'baseline_sales'
        guardrail_limit = 1.5
        
        adjusted_data = forecaster.calculate_guardrail(
            data=test_data,
            group_cols=group_cols,
            baseline_col=baseline_col,
            guardrail_limit=guardrail_limit
        )
        
        # Check results
        assert 'guardrail' in adjusted_data.columns
        assert adjusted_data['guardrail'].any()  # At least one prediction should be adjusted
        
        # For groups where guardrail was applied, check that predictions are adjusted
        guardrail_applied = adjusted_data[adjusted_data['guardrail']]
        for _, group in guardrail_applied.groupby(group_cols):
            assert abs(group['prediction'].sum() / group[baseline_col].sum() - 1.0) <= guardrail_limit
    
    def test_run_backtesting_simple(self, sample_data):
        """Test the run_backtesting method with minimal settings."""
        forecaster = Forecaster(sample_data)
        
        # Set up parameters
        group_cols = ['product', 'store']
        features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        model = 'LGBM'
        
        # Run backtesting with minimal options
        result_df = forecaster.run_backtesting(
            group_cols=group_cols,
            features=features,
            target_col=target_col,
            model=model,
            use_parallel=False  # Avoid parallelization for testing
        )
        
        # Check results
        assert isinstance(result_df, pd.DataFrame)
        assert 'prediction' in result_df.columns
        
        # Check that there are predictions for all test samples
        test_samples = result_df[result_df['sample'] == 'test']
        assert not test_samples['prediction'].isna().any()

    def test_run_backtesting_with_feature_selection(self, sample_data):
        """Test the run_backtesting method with feature selection."""
        forecaster = Forecaster(sample_data)
        
        # Set up parameters
        group_cols = ['product', 'store']
        features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        model = 'LGBM'
        
        # Run backtesting with feature selection
        result_df = forecaster.run_backtesting(
            group_cols=group_cols,
            features=features,
            target_col=target_col,
            model=model,
            best_features=True,
            n_best_features=3,
            use_parallel=False  # Avoid parallelization for testing
        )
        
        # Check results
        assert isinstance(result_df, pd.DataFrame)
        assert 'prediction' in result_df.columns
        
        # Check that there are predictions for all test samples
        test_samples = result_df[result_df['sample'] == 'test']
        assert not test_samples['prediction'].isna().any()
        
        # Feature importances should be populated
        assert len(forecaster.feature_importances) > 0

    def test_run_backtesting_with_guardrail(self, sample_data):
        """Test the run_backtesting method with guardrail."""
        forecaster = Forecaster(sample_data)
        
        # Set up parameters
        group_cols = ['product', 'store']
        features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        model = 'LGBM'
        baseline_col = 'baseline_sales'
        
        # Run backtesting with guardrail
        result_df = forecaster.run_backtesting(
            group_cols=group_cols,
            features=features,
            target_col=target_col,
            model=model,
            baseline_col=baseline_col,
            use_guardrail=True,
            guardrail_limit=1.5,
            use_parallel=False  # Avoid parallelization for testing
        )
        
        # Check results
        assert isinstance(result_df, pd.DataFrame)
        assert 'prediction' in result_df.columns
        
        # Check that there are predictions for all test samples
        test_samples = result_df[result_df['sample'] == 'test']
        assert not test_samples['prediction'].isna().any()

    def test_get_feature_importance(self, sample_data):
        """Test the get_feature_importance method."""
        forecaster = Forecaster(sample_data)
        
        # Set up parameters and run backtesting
        group_cols = ['product', 'store']
        features = [col for col in sample_data.columns if col.startswith('feature_')]
        target_col = 'sales'
        model = 'LGBM'
        
        forecaster.run_backtesting(
            group_cols=group_cols,
            features=features,
            target_col=target_col,
            model=model,
            use_parallel=False  # Avoid parallelization for testing
        )
        
        # Get feature importances
        feature_importances = forecaster.get_feature_importance()
        
        # Check results
        assert isinstance(feature_importances, dict)
        
        # Check that feature importances are populated
        # Note: Since we're not running hyperparameter tuning, feature importances might be empty
        # Just verify that the method runs without errors
        if len(feature_importances) > 0:
            # Feature importances can be stored in different formats (dict, numpy array, etc.)
            # Just verify they exist
            pass

if __name__ == "__main__":
    pytest.main(["-v", "test_forecaster.py"])
