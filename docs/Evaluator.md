## Class: `Evaluator`

This class is designed to assess the performance of predictive models against actual sales data. It provides methods for calculating various error metrics, including RMSE, MAE, MAPE, WMAPE, and a custom metric. These metrics help quantify the accuracy of baseline and predicted sales, aiding in model evaluation and selection.

---

### 1. `__init__`

- **Description**: Initializes the `Evaluator` instance with the provided DataFrame and specifies the relevant columns for actual sales, baseline, and predictions.
- **Parameters**:
  - `df`: A pandas DataFrame containing the input data with sales, baseline, and predictions.
  - `actuals_col`: The column name for the actual sales data.
  - `baseline_col`: The column name for the baseline data.
  - `preds_cols`: A list of column names for the predictions data.

---

### 2. `_filter_test_data`

- **Description**: Filters the DataFrame to include only the test samples and non-zero actual sales values.
- **Returns**: A filtered DataFrame containing only the test samples and non-zero actual sales.

---

### 3. `_remove_nan`

- **Description**: Removes NaN values from the actuals and predictions arrays.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: Cleaned arrays of actuals and predictions without NaN values.

---

### 4. `calculate_rmse`

- **Description**: Calculates the Root Mean Square Error (RMSE) between actual sales and predictions.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: The calculated RMSE value.

---

### 5. `calculate_mape`

- **Description**: Calculates the Mean Absolute Percentage Error (MAPE) between actual sales and predictions.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: The calculated MAPE value.

---

### 6. `calculate_wmape`

- **Description**: Calculates the Weighted Mean Absolute Percentage Error (WMAPE) between actual sales and predictions.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: The calculated WMAPE value.

---

### 7. `calculate_mae`

- **Description**: Calculates the Mean Absolute Error (MAE) between actual sales and predictions.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: The calculated MAE value.

---

### 8. `calculate_custom_metric`

- **Description**: Calculates a custom metric based on the absolute error and overall error of predictions.
- **Parameters**:
  - `actuals`: Array of actual sales values.
  - `predictions`: Array of predicted or baseline sales values.
- **Returns**: The computed custom metric score.

---

### 9. `evaluate`

- **Description**: Evaluates and computes metrics (RMSE, MAE, MAPE, WMAPE, and custom metric) for the baseline and all prediction models.
- **Returns**: A dictionary containing metrics for the baseline and all predictions.

---

### 10. `create_metric_table`

- **Description**: Creates a DataFrame summarizing the evaluation metrics, rounding the values to two decimal places.
- **Returns**: A pandas DataFrame containing the rounded metrics for baseline and all predictions.

---