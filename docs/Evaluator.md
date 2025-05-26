# Evaluator Class Documentation

## Overview

The `Evaluator` class is a critical component of the Forecaster pipeline that provides comprehensive model evaluation capabilities. It calculates and compares various performance metrics between actual values and predictions from different models. The class supports multiple standard metrics (RMSE, MAE, MAPE, WMAPE) and a custom metric, with built-in handling for common data issues like NaN values and zero divisions.

This evaluation framework allows users to:
1. Compare baseline models against advanced forecasting models
2. Analyze performance across different time horizons or groups
3. Generate summary tables of performance metrics
4. Handle edge cases gracefully (zeros, NaN values)

---

### 1. `__init__`

- **Description**: Initializes the `Evaluator` instance with the provided DataFrame and specifies the relevant columns for actual values, baseline, and predictions. The class makes a copy of the input DataFrame to avoid modifying the original data.
- **Parameters**:
  - `df` (pd.DataFrame): The input DataFrame containing actual values, baseline predictions, and model predictions.
  - `actuals_col` (str): The column name for the actual values (e.g., 'sales', 'inventory').
  - `baseline_col` (str): The column name for the baseline predictions (e.g., 'baseline_sales').
  - `preds_cols` (list): A list of column names for the model predictions (e.g., ['prediction_lgbm', 'prediction_lr']).

---

### 2. `_filter_test_data`

- **Description**: Filters the DataFrame to include only the test samples and non-zero actual values. This internal method is used by other evaluation functions to ensure metrics are calculated only on relevant test data.
- **Implementation**: Applies a filter for rows where 'sample' column equals 'test' and the actuals column has non-zero values.
- **Returns**: A filtered DataFrame containing only the test samples with non-zero actual values.

---

### 3. `_remove_nan`

- **Description**: Removes NaN values from the actuals and predictions arrays to ensure metric calculations are performed only on valid pairs of values. This internal method is used by all metric calculation functions.
- **Implementation**: Creates a mask identifying positions where neither the actuals nor predictions arrays contain NaN values, then applies this mask to filter both arrays.
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: Tuple containing cleaned arrays of actuals and predictions with all NaN values removed.

---

### 4. `calculate_rmse`

- **Description**: Calculates the Root Mean Square Error (RMSE) between actual values and predictions. RMSE is a standard metric that measures the square root of the average squared differences between predicted and actual values. It gives higher weight to larger errors.
- **Formula**: RMSE = sqrt(mean((actuals - predictions)²))
- **Implementation**: First removes NaN values, then calculates the mean squared error and takes the square root.
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: The calculated RMSE value as a float.

---

### 5. `calculate_mape`

- **Description**: Calculates the Mean Absolute Percentage Error (MAPE) between actual values and predictions. MAPE expresses accuracy as a percentage of error, showing the average absolute percent difference between predicted and actual values.
- **Formula**: MAPE = mean(abs((actuals - predictions) / actuals)) * 100
- **Implementation**: First removes NaN values, then filters out zero values in actuals to avoid division by zero. If no non-zero actuals remain, returns NaN.
- **Edge Cases**: 
  - Returns NaN if all actual values are zero
  - Excludes data points where actual value is zero (to avoid division by zero)
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: The calculated MAPE value as a percentage (float), or NaN if all actuals are zero.

---

### 6. `calculate_wmape`

- **Description**: Calculates the Weighted Mean Absolute Percentage Error (WMAPE) between actual values and predictions. Unlike MAPE, which treats all errors equally, WMAPE weights the errors by the magnitude of the actual values, giving more importance to errors on larger values.
- **Formula**: WMAPE = sum(abs(actuals - predictions)) / sum(actuals) * 100
- **Implementation**: First removes NaN values, then calculates the sum of absolute errors divided by the sum of actual values, multiplied by 100. If total actuals sum to zero, returns NaN.
- **Edge Cases**: 
  - Returns NaN if the sum of actual values is zero
  - Generally more robust than MAPE for datasets with occasional zero or near-zero values
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: The calculated WMAPE value as a percentage (float), or NaN if sum of actuals is zero.

---

### 7. `calculate_mae`

- **Description**: Calculates the Mean Absolute Error (MAE) between actual values and predictions. MAE measures the average absolute difference between predicted and actual values, providing a straightforward measure of error in the same units as the original data.
- **Formula**: MAE = mean(abs(actuals - predictions))
- **Implementation**: First removes NaN values, then calculates the mean of absolute differences.
- **Advantages**: 
  - Less sensitive to outliers than RMSE
  - Provides error in the same units as the data (unlike MAPE or WMAPE)
  - Works well with zero values (unlike MAPE)
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: The calculated MAE value as a float.

---

### 8. `calculate_custom_metric`

- **Description**: Calculates a custom metric that accounts for both the magnitude of errors and the directional bias in predictions. This metric combines absolute error and overall error, providing a comprehensive assessment that penalizes both large individual errors and systematic bias.
- **Formula**: Custom Metric = (sum(abs(predictions - actuals)) + abs(sum(predictions - actuals))) / sum(actuals) * 100
- **Implementation**: First removes NaN values, then calculates both the sum of absolute errors and the absolute value of the overall error (which captures bias direction). These are combined and normalized by the total of actual values, then multiplied by 100 to express as a percentage.
- **Edge Cases**: 
  - Returns NaN if the sum of actual values is zero
  - Penalizes both overestimation and underestimation
- **Parameters**:
  - `actuals` (numpy.ndarray): Array of actual values.
  - `predictions` (numpy.ndarray): Array of predicted or baseline values.
- **Returns**: The computed custom metric score as a percentage (float), or NaN if sum of actuals is zero.

---

### 9. `evaluate`

- **Description**: Evaluates and computes all supported metrics (RMSE, MAE, MAPE, WMAPE, and custom metric) for the baseline and all prediction models. This method provides a comprehensive assessment of model performance across multiple error measures.
- **Implementation**: 
  - Filters the data to include only test samples with non-zero actual values
  - Calculates each metric for the baseline predictions
  - Calculates each metric for all model predictions specified in `preds_cols`
  - Organizes results into a nested dictionary structure for easy access
- **Returns**: A dictionary with the structure `{metric_name: {column_name: value}}`, containing calculated metrics for the baseline and all prediction models. For example: 
  ```
  {
    'RMSE': {'baseline_sales': 10.5, 'prediction_model1': 8.2, 'prediction_model2': 7.5},
    'MAE': {'baseline_sales': 8.3, 'prediction_model1': 6.7, 'prediction_model2': 6.1},
    ...
  }
  ```

---

### 10. `create_metric_table`

- **Description**: Creates a well-formatted DataFrame summarizing all evaluation metrics for easy comparison between baseline and prediction models. All values are rounded to two decimal places for readability.
- **Implementation**: 
  - Calls the `evaluate` method to get all metrics
  - Converts the nested dictionary into a pandas DataFrame
  - Rounds all numeric values to 2 decimal places
- **Returns**: A pandas DataFrame with metrics as columns and models (baseline and predictions) as rows. Example format:
  ```
                    RMSE    MAE   MAPE  WMAPE  Custom Metric
  baseline_sales    10.50   8.30  12.50  10.20          15.30
  prediction_model1  8.20   6.70   9.80   8.50          12.10
  prediction_model2  7.50   6.10   8.90   7.80          11.40
  ```

---

### 11. `_calculate_metric`

- **Description**: An internal helper function that calculates a specified metric based on the actual and predicted values. This method centralizes the logic for metric calculation, allowing other methods to specify metrics by name rather than duplicating calculation logic.
- **Implementation**: 
  - Uses a conditional structure to direct the calculation to the appropriate metric function
  - Validates that the requested metric is supported
- **Parameters**:
  - `metric_name` (str): The name of the metric to calculate (one of 'RMSE', 'MAE', 'MAPE', 'WMAPE', or 'Custom Metric').
  - `actuals` (numpy.ndarray): The actual values to compare against.
  - `predictions` (numpy.ndarray): The predicted values generated by the model.
- **Returns**: The calculated metric value based on the specified metric name.
- **Raises**: 
  - `ValueError`: If an unknown metric name is provided.

---

### 12. `calculate_grouped_metric`

- **Description**: Calculates a specified metric (e.g., RMSE, MAE) grouped by a specified column, allowing analysis of model performance across different categories or time horizons. This is particularly useful for understanding how model performance varies by forecast lag, product category, store type, or other grouping variables.
- **Implementation**: 
  - Filters test data with non-zero actual values
  - Optionally applies a filter to include only specific group values
  - Groups the data by the specified column
  - For each group, calculates the specified metric for baseline and all prediction models
  - Organizes results into a transposed DataFrame with models as rows and group values as columns
- **Parameters**:
  - `metric_name` (str): The name of the metric to calculate (one of 'RMSE', 'MAE', 'MAPE', 'WMAPE', or 'Custom Metric').
  - `group_col` (str): The column name used for grouping (e.g., 'fcst_lag', 'product_category', 'store_type').
  - `group_filter` (list, optional): A list of values to filter groups. If provided, only groups with values in this list will be included in the analysis.
- **Returns**: A pandas DataFrame with models as rows (baseline and prediction models) and unique values of the group column as columns. Each cell contains the calculated metric value for that model-group combination.
- **Example Format**:
  ```
                    1       2       3       4       5
  Baseline         12.50   14.20   15.80   17.30   20.10
  prediction_model1  9.80   10.50   12.20   14.50   16.80
  prediction_model2  8.90    9.70   11.50   13.20   15.40
  ```
  where column names 1-5 might represent forecast lag values.

---