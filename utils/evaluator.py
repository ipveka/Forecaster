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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.model_selection import TimeSeriesSplit
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import mutual_info_regression

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

# Evaluator class

class Evaluator:
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

    def _filter_test_data(self):
        """
        Filters the dataframe to include only the test samples and non-zero actuals.

        :return: Filtered DataFrame with only test samples and non-zero actual sales
        """
        return self.df[(self.df['sample'] == 'test') & (self.df[self.actuals_col] != 0)]

    def _remove_nan(self, actuals, predictions):
        """
        Remove NaN values from both actuals and predictions.

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: Cleaned actuals and predictions arrays
        """
        mask = ~np.isnan(actuals) & ~np.isnan(predictions)
        return actuals[mask], predictions[mask]

    def calculate_rmse(self, actuals, predictions):
        """
        Calculate Root Mean Square Error (RMSE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: RMSE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.sqrt(mean_squared_error(actuals, predictions))

    def calculate_mape(self, actuals, predictions):
        """
        Calculate Mean Absolute Percentage Error (MAPE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: MAPE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.mean(np.abs((actuals - predictions) / actuals)) * 100

    def calculate_wmape(self, actuals, predictions):
        """
        Calculate Weighted Mean Absolute Percentage Error (WMAPE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: WMAPE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.sum(np.abs(actuals - predictions)) / np.sum(actuals) * 100

    def calculate_mae(self, actuals, predictions):
        """
        Calculate Mean Absolute Error (MAE).

        :param actuals: Array of actual sales values
        :param predictions: Array of predicted or baseline sales values
        :return: MAE value
        """
        actuals, predictions = self._remove_nan(actuals, predictions)
        return np.mean(np.abs(actuals - predictions))

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
        if total_objective != 0:
            score /= total_objective
        else:
            score = np.nan

        return score * 100

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