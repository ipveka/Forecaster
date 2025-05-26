"""
Unit tests for the Evaluator class focusing on evaluation metrics and functionality.
"""
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import the Evaluator class
from utils.evaluator import Evaluator

class TestEvaluator:
    """Test suite for Evaluator class."""

    @pytest.fixture
    def sample_data(self):
        """Create a sample dataset for testing the Evaluator class."""
        # Create date range
        start_date = datetime(2022, 1, 1)
        dates = [start_date + timedelta(days=i*7) for i in range(12)]  # Weekly data for 3 months
        
        # Create product IDs
        product_ids = ['P001', 'P002']
        
        # Create store IDs
        store_ids = ['S01', 'S02']
        
        # Create a list to hold all records
        records = []
        
        for date in dates:
            for product_id in product_ids:
                for store_id in store_ids:
                    # Generate some sample data
                    # Base sales value with some noise
                    actual_sales = float(np.random.normal(100, 20))
                    
                    # Create baseline and two prediction models
                    # Baseline is less accurate than predictions
                    baseline_sales = actual_sales * np.random.normal(1.1, 0.2)  # Baseline with ~10% error
                    pred_model1 = actual_sales * np.random.normal(1.05, 0.1)    # Model 1 with ~5% error
                    pred_model2 = actual_sales * np.random.normal(1.02, 0.05)   # Model 2 with ~2% error
                    
                    # Add a forecasting lag column for grouped metric testing
                    fcst_lag = (date.month - 1) % 4 + 1  # Lag will be 1, 2, 3, or 4
                    
                    # Create a record
                    record = {
                        'date': date,
                        'product': product_id,
                        'store': store_id,
                        'actual_sales': actual_sales,
                        'baseline_sales': baseline_sales,
                        'prediction_model1': pred_model1,
                        'prediction_model2': pred_model2,
                        'fcst_lag': fcst_lag,
                        'sample': 'test' if date.month > 1 else 'train'  # First month as training, rest as test
                    }
                    records.append(record)
        
        # Create some records with zero sales and NaN values for edge case testing
        edge_cases = [
            {
                'date': start_date + timedelta(days=90),
                'product': 'P003',
                'store': 'S03',
                'actual_sales': 0.0,  # Zero sales
                'baseline_sales': 10.0,
                'prediction_model1': 5.0,
                'prediction_model2': 2.0,
                'fcst_lag': 1,
                'sample': 'test'
            },
            {
                'date': start_date + timedelta(days=97),
                'product': 'P003',
                'store': 'S03',
                'actual_sales': np.nan,  # NaN actual sales
                'baseline_sales': 15.0,
                'prediction_model1': 10.0,
                'prediction_model2': 12.0,
                'fcst_lag': 2,
                'sample': 'test'
            },
            {
                'date': start_date + timedelta(days=104),
                'product': 'P003',
                'store': 'S03',
                'actual_sales': 20.0,
                'baseline_sales': np.nan,  # NaN baseline
                'prediction_model1': 18.0,
                'prediction_model2': 21.0,
                'fcst_lag': 3,
                'sample': 'test'
            }
        ]
        records.extend(edge_cases)
        
        # Create DataFrame from the records
        df = pd.DataFrame(records)
        return df

    def test_init(self, sample_data):
        """Test initialization of Evaluator class."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Check that df was correctly copied
        assert isinstance(evaluator.df, pd.DataFrame)
        assert len(evaluator.df) == len(sample_data)
        
        # Check that column names were correctly stored
        assert evaluator.actuals_col == actuals_col
        assert evaluator.baseline_col == baseline_col
        assert evaluator.preds_cols == preds_cols

    def test_filter_test_data(self, sample_data):
        """Test _filter_test_data method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Get filtered data
        filtered_data = evaluator._filter_test_data()
        
        # Check that only test samples are included
        assert all(filtered_data['sample'] == 'test')
        
        # Check that zero sales are excluded
        assert all(filtered_data[actuals_col] != 0)
        
        # Check expected number of rows
        expected_test_rows = len(sample_data[(sample_data['sample'] == 'test') & (sample_data[actuals_col] != 0)])
        assert len(filtered_data) == expected_test_rows

    def test_remove_nan(self, sample_data):
        """Test _remove_nan method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Create arrays with NaN values
        actuals = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        predictions = np.array([1.1, np.nan, 3.1, 4.1, 5.1])
        
        # Remove NaN values
        clean_actuals, clean_predictions = evaluator._remove_nan(actuals, predictions)
        
        # Check that NaN values are removed from both arrays
        assert len(clean_actuals) == 3  # Should only have values at indices 0, 3, and 4
        assert len(clean_predictions) == 3
        assert np.array_equal(clean_actuals, np.array([1.0, 4.0, 5.0]))
        assert np.array_equal(clean_predictions, np.array([1.1, 4.1, 5.1]))

    def test_calculate_rmse(self, sample_data):
        """Test calculate_rmse method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Define test data
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        # Expected RMSE = sqrt(mean((actuals - predictions)^2))
        expected_rmse = np.sqrt(np.mean(np.square(actuals - predictions)))
        
        # Calculate RMSE
        rmse = evaluator.calculate_rmse(actuals, predictions)
        
        # Check result
        assert rmse == pytest.approx(expected_rmse)

    def test_calculate_mape(self, sample_data):
        """Test calculate_mape method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Define test data (without zeros to avoid division by zero)
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        # Expected MAPE = mean(abs((actuals - predictions) / actuals)) * 100
        expected_mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        
        # Calculate MAPE
        mape = evaluator.calculate_mape(actuals, predictions)
        
        # Check result
        assert mape == pytest.approx(expected_mape)
        
        # Test with zeros in actuals (should return np.nan)
        actuals_with_zeros = np.array([0.0, 20.0, 30.0, 40.0, 50.0])
        predictions_for_zeros = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        mape_with_zeros = evaluator.calculate_mape(actuals_with_zeros, predictions_for_zeros)
        
        # Check that non-zero values are still used
        expected_mape_with_zeros = np.mean(np.abs((actuals_with_zeros[1:] - predictions_for_zeros[1:]) / actuals_with_zeros[1:])) * 100
        assert mape_with_zeros == pytest.approx(expected_mape_with_zeros)
        
        # Test with all zeros in actuals (should return np.nan)
        all_zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        predictions_for_all_zeros = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        mape_all_zeros = evaluator.calculate_mape(all_zeros, predictions_for_all_zeros)
        
        # Check that np.nan is returned when all actuals are zero
        assert np.isnan(mape_all_zeros)

    def test_calculate_wmape(self, sample_data):
        """Test calculate_wmape method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Define test data
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        # Expected WMAPE = sum(abs(actuals - predictions)) / sum(actuals) * 100
        expected_wmape = np.sum(np.abs(actuals - predictions)) / np.sum(actuals) * 100
        
        # Calculate WMAPE
        wmape = evaluator.calculate_wmape(actuals, predictions)
        
        # Check result
        assert wmape == pytest.approx(expected_wmape)
        
        # Test with zeros (sum of actuals is not zero)
        actuals_with_zeros = np.array([0.0, 20.0, 30.0, 40.0, 50.0])
        predictions_for_zeros = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        wmape_with_zeros = evaluator.calculate_wmape(actuals_with_zeros, predictions_for_zeros)
        
        expected_wmape_with_zeros = np.sum(np.abs(actuals_with_zeros - predictions_for_zeros)) / np.sum(actuals_with_zeros) * 100
        assert wmape_with_zeros == pytest.approx(expected_wmape_with_zeros)
        
        # Test with all zeros in actuals (should return np.nan)
        all_zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        predictions_for_all_zeros = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        wmape_all_zeros = evaluator.calculate_wmape(all_zeros, predictions_for_all_zeros)
        
        # Check that np.nan is returned when sum of actuals is zero
        assert np.isnan(wmape_all_zeros)

    def test_calculate_mae(self, sample_data):
        """Test calculate_mae method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Define test data
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        # Expected MAE = mean(abs(actuals - predictions))
        expected_mae = np.mean(np.abs(actuals - predictions))
        
        # Calculate MAE
        mae = evaluator.calculate_mae(actuals, predictions)
        
        # Check result
        assert mae == pytest.approx(expected_mae)

    def test_calculate_custom_metric(self, sample_data):
        """Test calculate_custom_metric method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Define test data
        actuals = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        predictions = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        # Expected custom metric = (sum(abs(predictions - actuals)) + abs(sum(predictions - actuals))) / sum(actuals) * 100
        abs_err = np.sum(np.abs(predictions - actuals))
        err = np.sum(predictions - actuals)
        expected_custom_metric = (abs_err + abs(err)) / np.sum(actuals) * 100
        
        # Calculate custom metric
        custom_metric = evaluator.calculate_custom_metric(actuals, predictions)
        
        # Check result
        assert custom_metric == pytest.approx(expected_custom_metric)
        
        # Test with all zeros in actuals (should return np.nan)
        all_zeros = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        predictions_for_all_zeros = np.array([12.0, 18.0, 33.0, 36.0, 55.0])
        
        custom_metric_all_zeros = evaluator.calculate_custom_metric(all_zeros, predictions_for_all_zeros)
        
        # Check that np.nan is returned when sum of actuals is zero
        assert np.isnan(custom_metric_all_zeros)

    def test_evaluate(self, sample_data):
        """Test evaluate method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Run evaluation
        metrics = evaluator.evaluate()
        
        # Check structure of metrics dictionary
        assert isinstance(metrics, dict)
        assert all(metric in metrics for metric in ['RMSE', 'MAE', 'MAPE', 'WMAPE', 'Custom Metric'])
        
        # Check that metrics are calculated for baseline and all prediction models
        for metric in metrics:
            assert baseline_col in metrics[metric]
            for pred_col in preds_cols:
                assert pred_col in metrics[metric]
                
            # Check that metrics are numeric values (not NaN for test data)
            assert not np.isnan(metrics[metric][baseline_col])
            for pred_col in preds_cols:
                assert not np.isnan(metrics[metric][pred_col])

    def test_create_metric_table(self, sample_data):
        """Test create_metric_table method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Create metric table
        metric_table = evaluator.create_metric_table()
        
        # Check that the result is a DataFrame
        assert isinstance(metric_table, pd.DataFrame)
        
        # Check columns of the DataFrame (should be the metrics)
        assert all(metric in metric_table.columns for metric in ['RMSE', 'MAE', 'MAPE', 'WMAPE', 'Custom Metric'])
        
        # Check rows of the DataFrame (should be baseline and prediction models)
        assert baseline_col in metric_table.index
        for pred_col in preds_cols:
            assert pred_col in metric_table.index
            
        # Check that all values in the table are rounded to 2 decimal places
        for col in metric_table.columns:
            for idx in metric_table.index:
                # Check that the value is a number (not NaN)
                assert not np.isnan(metric_table.loc[idx, col])
                
                # Check that the value is rounded to 2 decimal places
                value = metric_table.loc[idx, col]
                assert value == pytest.approx(round(value, 2))

    def test_calculate_grouped_metric(self, sample_data):
        """Test calculate_grouped_metric method."""
        # Define parameters
        actuals_col = 'actual_sales'
        baseline_col = 'baseline_sales'
        preds_cols = ['prediction_model1', 'prediction_model2']
        
        # Initialize Evaluator
        evaluator = Evaluator(sample_data, actuals_col, baseline_col, preds_cols)
        
        # Calculate grouped metrics by fcst_lag
        grouped_rmse = evaluator.calculate_grouped_metric('RMSE', 'fcst_lag')
        
        # Check that the result is a DataFrame
        assert isinstance(grouped_rmse, pd.DataFrame)
        
        # Check columns of the DataFrame (should be the unique fcst_lag values)
        test_data = sample_data[(sample_data['sample'] == 'test') & (sample_data[actuals_col] != 0)]
        unique_lags = test_data['fcst_lag'].unique()
        for lag in unique_lags:
            assert lag in grouped_rmse.columns
            
        # Check rows of the DataFrame (should be 'Baseline' and prediction models)
        assert 'Baseline' in grouped_rmse.index
        for pred_col in preds_cols:
            assert pred_col in grouped_rmse.index
            
        # Test with group filter
        # Get the actual available lags in the filtered test data
        test_data = sample_data[(sample_data['sample'] == 'test') & (sample_data[actuals_col] != 0)]
        available_lags = sorted(test_data['fcst_lag'].unique())
        
        # Use the first two available lags for filtering
        if len(available_lags) >= 2:
            filter_lags = available_lags[:2]
            filtered_grouped_rmse = evaluator.calculate_grouped_metric('RMSE', 'fcst_lag', group_filter=filter_lags)
            
            # Check that only filtered lags are included
            assert set(filtered_grouped_rmse.columns) == set(filter_lags)
        
        # Check other metrics
        for metric in ['MAE', 'MAPE', 'WMAPE', 'Custom Metric']:
            grouped_metric = evaluator.calculate_grouped_metric(metric, 'fcst_lag')
            
            # Check that the result is a DataFrame
            assert isinstance(grouped_metric, pd.DataFrame)
            
            # Check that all columns and rows are as expected
            for lag in unique_lags:
                assert lag in grouped_metric.columns
            assert 'Baseline' in grouped_metric.index
            for pred_col in preds_cols:
                assert pred_col in grouped_metric.index

if __name__ == "__main__":
    pytest.main(["-v", "test_evaluator.py"])
