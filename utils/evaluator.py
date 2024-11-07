# General libraries
import pandas as pd
import numpy as np
import warnings
import lightgbm
import gc
import os

# Plots
from matplotlib import pyplot as plt

# Sklearn
from sklearn.metrics import mean_squared_error

# Forecaster
from lightgbm import LGBMRegressor

# Multiprocessing
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from functools import partial

# Plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython import display
from time import sleep
from math import ceil

# Cuda
import torch

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Evaluator class
class Evaluator:
    # Init
    def __init__(self, df, actuals_col, baseline_col, preds_cols):
        """
        Initialize the Evaluator with the dataframe that contains the data.

        :param df: A pandas DataFrame containing the input data with sales, baseline, and predictions.
        :param baseline_col: The column name for the baseline data.
        :param preds_cols: A list of column names for the predictions data.
        :param actuals_col: The column name for the actual sales data.
        """
        self.df = df.copy()
        self.baseline_col = baseline_col
        self.preds_cols = preds_cols
        self.actuals_col = actuals_col

    # Filter test
    def _filter_test_data(self):
        """
        Filters the dataframe to include only the test samples and non-zero actuals.

        :return: Filtered DataFrame with only test samples and non-zero actual sales
        """
        return self.df[(self.df['sample'] == 'test') & (self.df[self.actuals_col] != 0)]

    # Remove nan
    def _remove_nan(self, actuals, predictions):
        """
        Remove NaN values from both actuals and predictions.

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: Cleaned actuals and predictions arrays
        """
        mask = ~np.isnan(actuals) & ~np.isnan(predictions)
        return actuals[mask], predictions[mask]

    # Calculate RMSE
    def calculate_rmse(self, actuals, predictions):
        """
        Calculate Root Mean Square Error (RMSE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: RMSE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.sqrt(mean_squared_error(actuals, predictions))

    # Calculate MAPE
    def calculate_mape(self, actuals, predictions):
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: MAPE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        
        # Avoid division by zero by filtering out zero values in actuals
        non_zero_actuals = actuals != 0
        if np.any(non_zero_actuals):
            return np.mean(np.abs((actuals[non_zero_actuals] - predictions[non_zero_actuals]) / actuals[non_zero_actuals])) * 100
        else:
            return np.nan
        
    # Calculate WMAPE
    def calculate_wmape(self, actuals, predictions):
        """
        Calculate Weighted Mean Absolute Percentage Error (WMAPE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: WMAPE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        total_actuals = np.sum(actuals)
        
        # Avoid division by zero by checking if total actuals sum to zero
        if total_actuals != 0:
            return np.sum(np.abs(actuals - predictions)) / total_actuals * 100
        else:
            return np.nan

    # Calculate MAE
    def calculate_mae(self, actuals, predictions):
        """
        Calculate Mean Absolute Error (MAE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: MAE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.mean(np.abs(actuals - predictions))

    # Calculate custom metric
    def calculate_custom_metric(self, actuals, predictions):
        """
        Calculate the custom metric based on absolute error and overall error.

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: Custom metric score
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        abs_err = np.sum(np.abs(predictions - actuals))
        err = np.sum(predictions - actuals)
        score = abs_err + abs(err)

        total_objective = np.sum(actuals)
        
        # Avoid division by zero for custom metric
        if total_objective != 0:
            score /= total_objective
        else:
            score = np.nan

        return score * 100

    # Evaluate
    def evaluate(self):
        """
        Evaluate RMSE, MAE, MAPE, WMAPE, and custom metric for baseline and all prediction models.

        :return: A dictionary containing metrics for baseline and all predictions
        """
        test_data = self._filter_test_data()

        actuals = test_data[self.actuals_col].values
        baseline = test_data[self.baseline_col].values

        # Initialize metrics dictionary using the baseline column name
        metrics = {metric: {self.baseline_col: None} for metric in ['RMSE', 'MAE', 'MAPE', 'WMAPE', 'Custom Metric']}

        # Calculate metrics for baseline
        for metric in metrics:
            if metric == 'RMSE':
                metrics[metric][self.baseline_col] = self.calculate_rmse(actuals, baseline)
            elif metric == 'MAE':
                metrics[metric][self.baseline_col] = self.calculate_mae(actuals, baseline)
            elif metric == 'MAPE':
                metrics[metric][self.baseline_col] = self.calculate_mape(actuals, baseline)
            elif metric == 'WMAPE':
                metrics[metric][self.baseline_col] = self.calculate_wmape(actuals, baseline)
            elif metric == 'Custom Metric':
                metrics[metric][self.baseline_col] = self.calculate_custom_metric(actuals, baseline)

        # Calculate metrics for each prediction model
        for pred_col in self.preds_cols:
            predictions = test_data[pred_col].values

            for metric in metrics:
                if metric == 'RMSE':
                    metrics[metric][pred_col] = self.calculate_rmse(actuals, predictions)
                elif metric == 'MAE':
                    metrics[metric][pred_col] = self.calculate_mae(actuals, predictions)
                elif metric == 'MAPE':
                    metrics[metric][pred_col] = self.calculate_mape(actuals, predictions)
                elif metric == 'WMAPE':
                    metrics[metric][pred_col] = self.calculate_wmape(actuals, predictions)
                elif metric == 'Custom Metric':
                    metrics[metric][pred_col] = self.calculate_custom_metric(actuals, predictions)

        return metrics

    # Create metrics table
    def create_metric_table(self):
        """
        Create a DataFrame for the evaluation metrics, with values rounded to 2 decimal places.

        :return: A pandas DataFrame summarizing the metrics for baseline and all predictions
        """
        metrics = self.evaluate()

        # Creating a DataFrame from the metrics dictionary
        metric_table = pd.DataFrame(metrics)

        # Round all numeric values to 2 decimal places
        metric_table = metric_table.round(2)

        return metric_table
    
    # Calculate metrics
    def _calculate_metric(self, metric_name, actuals, predictions):
        """
        Helper function to calculate the specified metric.

        :param metric_name: The name of the metric to calculate
        :param actuals: The actual values
        :param predictions: The predicted values
        :return: The calculated metric value
        """
        if metric_name == 'RMSE':
            return self.calculate_rmse(actuals, predictions)
        elif metric_name == 'MAE':
            return self.calculate_mae(actuals, predictions)
        elif metric_name == 'MAPE':
            return self.calculate_mape(actuals, predictions)
        elif metric_name == 'WMAPE':
            return self.calculate_wmape(actuals, predictions)
        elif metric_name == 'Custom Metric':
            return self.calculate_custom_metric(actuals, predictions)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    # Calculate grouped metrics
    def calculate_grouped_metric(self, metric_name, group_col, group_filter=None):
        """
        Calculate the specified metric grouped by the specified column,
        including baseline and all prediction models. Optionally filter 
        groups based on a specified range or list of values.

        :param metric_name: The name of the metric to calculate (e.g., 'RMSE', 'MAE', etc.)
        :param group_col: The column name to group by (e.g., 'fcst_lag')
        :param group_filter: Optional; a range or list of values to filter groups (e.g., range(1, 14))
        :return: A pandas DataFrame summarizing the metric for each prediction column and baseline by group_col
        """
        # Filter test data
        test_data = self._filter_test_data()
        actuals = test_data[self.actuals_col].values

        # Apply group filter if provided
        if group_filter is not None:
            test_data = test_data[test_data[group_col].isin(group_filter)]

        # Initialize a list to hold the metric values
        metric_values = []

        # Group the data by the specified column
        grouped = test_data.groupby(group_col)

        # Calculate metrics for each group
        for group_value, group_data in grouped:
            # Get actuals for the current group
            group_actuals = group_data[self.actuals_col].values

            # Prepare a row for the current group
            current_row = {}

            # Calculate baseline metrics
            baseline_predictions = group_data[self.baseline_col].values
            current_row['Baseline'] = self._calculate_metric(metric_name, group_actuals, baseline_predictions)

            # Calculate metrics for each prediction column
            for pred_col in self.preds_cols:
                predictions = group_data[pred_col].values
                current_row[pred_col] = self._calculate_metric(metric_name, group_actuals, predictions)

            # Append the current row of metrics to the list
            metric_values.append(current_row)

        # Create a DataFrame from the metric values list
        result_df = pd.DataFrame(metric_values, index=grouped.groups.keys())

        # Set the index name to group_col
        result_df.index.name = group_col

        # Transpose to have models as rows and fcst_lag as columns
        result_df = result_df.transpose()

        return result_df