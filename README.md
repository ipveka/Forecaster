# ⚡ Forecaster

**Forecaster** is a time series forecasting project using **LightGBM** and other gradient boosting models. It streamlines the forecasting pipeline with dedicated classes for each stage, from data preparation to model evaluation, making it adaptable to various time series forecasting use cases.

This project uses as example data provided for the HP Supply Chain Optimization challenge at HackUPC 2023. You can see all related information here: https://www.kaggle.com/competitions/hp-supply-chain-optimization

---

## Models available

- **LightGBM Regressor (LGBM)**: A gradient boosting model optimized for speed and efficiency, often used for large datasets.
- **Random Forest Regressor (RF)**: An ensemble method that combines multiple decision trees for robust predictions.
- **Gradient Boosting Regressor (GBM)**: A boosting model that builds sequential trees, each correcting the errors of the previous ones.
- **AdaBoost Regressor (ADA)**: A boosting technique using weak learners, typically decision trees, to improve overall model performance.
- **Linear Regression (LR)**: A linear model often used as a baseline; performs well with simple relationships between features and target

---

## 🌟 Why Gradient Boosting models are great for Forecasting

Gradient boosting machine (GBM) models are powerful tools for time series forecasting. They excel in capturing complex patterns, handling a wide range of data types, and scaling efficiently to large datasets. **LightGBM** (Light Gradient Boosting Machine) stands out as a popular GBM variant due to its focus on speed, accuracy, and efficiency. Here’s a look into why GBMs are excellent for forecasting, with a focus on what makes LightGBM especially effective.

---

### 🌿 The Power of Gradient Boosting

At a high level, gradient boosting is an **ensemble learning** technique. It combines multiple weak learners, typically decision trees, in a sequential way to build a strong, predictive model. Here’s how it works:

- **Sequential Model Improvement**: Gradient boosting builds models iteratively, each new model improving on the errors of the previous ones. This allows it to refine predictions progressively, enhancing accuracy with each step.
- **Focus on High-Error Instances**: By assigning higher weights to instances with larger prediction errors, GBM models excel at capturing hard-to-model data patterns, making them highly suitable for **nonlinear and seasonal relationships** often present in time series data.
- **Adaptability to Diverse Data**: GBMs handle various data types (continuous, categorical, and even missing data) with minimal preprocessing, adapting well to diverse real-world datasets.

---

### 🌿 Best in class: LigthGBM from Microsoft

LightGBM is a specialized, high-performance GBM variant that accelerates training and boosts efficiency without compromising accuracy. Its unique approach to tree growth and data processing makes it especially well-suited for forecasting:

- **Leaf-Wise Tree Growth**: Unlike other GBMs that grow trees level-wise, LightGBM uses **leaf-wise growth**, adding nodes to the leaf with the highest potential error reduction. This yields **deeper, more accurate trees**, optimizing model performance.
- **Histogram-Based Learning**: To reduce computation, LightGBM bins continuous values into discrete groups, allowing it to process large datasets efficiently while still capturing important trends and patterns.
- **Built-In Categorical Handling**: LightGBM natively handles categorical variables, such as days of the week or seasons, without the need for preprocessing (e.g., one-hot encoding), making it easier to use with time series data.

---

### ⏱ Advantages of LightGBM for Time Series Forecasting

Here’s why LightGBM is a top choice for time series forecasting:

- **Scalability for Large Datasets**  
LightGBM is designed to handle large, high-dimensional datasets, such as those often encountered in time series forecasting with many features (e.g., lags, trends, and seasonal variables). Its **optimized memory usage and fast training** make it ideal for production-level forecasting tasks with millions of rows.

- **Pattern and Trend Recognition**  
Tree-based models, like those in LightGBM, inherently capture **nonlinear relationships** and complex dependencies. This structure makes it effective in capturing time series patterns, such as seasonality and trend changes, often leading to highly accurate forecasts.

- **Robustness to Missing and Noisy Data**  
Real-world time series data often contain missing values and noise. LightGBM’s handling of missing data within the model, along with its ability to **deal with outliers**, ensures robustness, preventing these issues from skewing predictions.

- **Flexibility for Grouped and Segmented Forecasting**  
In many forecasting scenarios, there’s a need to create separate forecasts for different groups or segments (e.g., regions, products, or customer segments). LightGBM’s structure supports **segmented modeling**, allowing the creation of multiple, tailored forecasts in a single model run, ideal for multi-series forecasting tasks.

- **Parallelized and GPU-Compatible**  
LightGBM’s parallelization capabilities and GPU support make it extremely efficient, allowing rapid iteration and tuning. This is especially beneficial for time series forecasting, where models may need frequent retraining or adjustments as new data becomes available.

### 🧠 When to Use LightGBM for Forecasting

LightGBM is especially effective for:

- **High-Dimensional Forecasting**: When the dataset includes many features (e.g., weather, holiday flags, external drivers) that may impact future values.
- **Complex Patterns**: Ideal when the series exhibits strong seasonal or trend components and nonlinear relationships.
- **Multi-Series Forecasting**: When multiple series need to be forecasted in tandem (e.g., product or location-specific demand).

---

## 🌟 Highlights

The `Forecaster` project offers several classes that help create features to boost accuracy and efficiency in forecasting:

- **Backtesting Capabilities** 🧭: Simulates real-world model performance by evaluating it over time.
- **Automatic Best Feature Selection** 🎯: Optimizes feature relevance automatically for enhanced model performance.
- **Outlier Detection and Treatment** 🚨: Identifies and reduces the impact of outliers for more robust forecasting.
- **Support for Grouped Data** 👥: Generates models for various data segments, tailoring forecasts to each.
- **Parallel and GPU Computing** 🚀: Accelerates training with parallel processing and GPU support (in development)

---

## 🔍 **Classes**

The project is structured around five primary classes, each responsible for a specific stage in the forecasting pipeline:

- **`DataPreparation`** 🛠️  
   This class prepares datasets by addressing missing values, transforming data types, and ensuring a model-ready structure. It includes individual functions that:
   - **Ensure data completeness** to prevent gaps in information.
   - **Smooth the target variable** to help reduce noise.
   - **Create a backtesting dataset** with varied cutoff values (train/test sets).
   - **Guarantee a full horizon** for all groups to support consistent forecasting.
   - **Orchestrator** The `run_data_preparation` function orchestrates all these steps, allowing users to perform comprehensive data preparation in one go.

- **`FeatureEngineering`** 🔍  
   Building on the outputs from `DataPreparation`, this class generates and refines features crucial for model performance, including:
   - **Encoding categorical data** for modeling.
   - **Creating period-based features** that capture trends.
   - **Adding calendar-based features** to incorporate seasonality.
   - **Calculating moving averages** and other statistics from numeric signals.
   - **Lag generation** to provide context from past data points.
   - **Coefficient of variation and group combinations**, representing statistical variability and complex group interactions.
   - **Cluster-based features** derived from average target values, period since inception
   - **Training weights calculation** used in the model to give more importance to recent data
   - **Orchestrator** The `run_feature_engineering` function orchestrates all feature engineering steps in a single operation.

- **`CreateBaselines`** 🧩  
   This class uses the outputs from `FeatureEngineering` to establish baseline models that serve as benchmarks, enabling users to evaluate LightGBM’s improvements over simpler models. It also generates simple forecasts for numeric signals.

- **`Forecaster`** 🔮  
   The core class for training the model, `Forecaster` efficiently manages the forecasting process. This class leverages the features generated by `FeatureEngineering` and `CreateBaselines` to produce forecasts at each cutoff. Key capabilities include:
   - **Setting training groups** to create one model for each relevant segments.
   - **Hyperparameter tuning** for optimal model performance.
   - **Feature selection** to enhance model efficiency.
   - **Outlier treatment** to minimize anomalies in predictions.
   - **Parallel and GPU computing support** for faster, more scalable processing.

- **`Evaluator`** 📊  
   This class is dedicated to evaluating model performance through essential metrics and visualizations, allowing users to interpret forecasting accuracy and reliability. It includes functionality to measure metric accuracy by lag, providing additional insights into prediction performance across time horizons.

---

## 📁 Project Structure

The repository is organized into several folders for easy navigation:

- **`data`**  
  Contains both raw and processed datasets. The processed datasets are ready for model training following data cleaning and feature engineering.

- **`docs`**  
  Comprehensive project documentation, detailing the classes, functions, architecture, and usage guidelines:
  - **`DataPreparation.md`**: Documentation for the DataPreparation class
  - **`FeatureEngineering.md`**: Documentation for the FeatureEngineering class
  - **`CreateBaselines.md`**: Documentation for the CreateBaselines class
  - **`Forecaster.md`**: Documentation for the Forecaster class
  - **`Evaluator.md`**: Documentation for the Evaluator class
  - **`Runner.md`**: Documentation for the Runner class, which orchestrates the entire pipeline
  - **`ForecasterUtils.md`**: Documentation for utility functions in the forecaster_utils module

- **`notebooks`**  
  Jupyter notebooks demonstrating the Forecaster workflow, including:

  - **`data_preparation`**: Walks through each step of data preparation, feature engineering, and baseline creation individually, allowing users to customize executions.
  - **`forecasting`**: Covers model evaluation, hyperparameter tuning, feature selection, and forecasting using `Forecaster` class, with insights on both training and validation processes.
  - **`runner`**: An end-to-end workflow notebook utilizing the `Runner` class to streamline the entire process, creating a complete forecasting pipeline.

- **`src`**  
  Contains the main executable scripts, including `run_forecaster.py` which demonstrates how to use the Runner class.

- **`utils`**  
  The utility folder holding the main classes, auxiliary functions, and plotting tools for the project.

---

## 🔄 Recent Improvements

### Parameter Standardization

The framework has been enhanced with a standardized parameter naming convention for clarity and consistency:

- **`dp_window_size`**: Window size for data preparation operations (formerly `ma_window_size`)
- **`fe_window_size`**: Window sizes for feature engineering calculations (formerly `window_sizes`)
- **`bs_window_size`**: Window size for baseline creation (formerly `window_size`)

This naming convention makes it clear which component each parameter belongs to, improving code readability and maintenance.

### Frequency-Based Parameter Management

The framework now intelligently manages parameters based on the detected frequency of the data:

- Automatic frequency detection when not explicitly provided
- Retrieval of optimal parameters from `get_frequency_params` based on the frequency
- Different parameter sets for daily ('D'), weekly ('W'), and monthly ('M') data

This ensures that appropriate parameters are used regardless of the time series frequency, enhancing adaptability across different datasets.

---

### 📈 **Example**  

Following the use case for HP Supply Chain Optimization challenge at HackUPC 2023, this project contains end to end notebook example (*runner*) using the data provided for the competition. The goal of this competition is to forecast the inventory level that HP will have for multiple products on a weekly basis.

## Files

- **Source**: https://www.kaggle.com/competitions/hp-supply-chain-optimization

- **train.csv**: The training dataset containing historical data with all fields filled in. This file is used to train models.

- **test.csv**: The test dataset, containing data for which predictions are needed. 

### Fields

- **id**: Unique identifier of the time series. Format: `yearweek-product_number`
    - Example: For the first week of 2023 and the product "1234567," the id would be `202301-1234567`.

- **date**: Date of record in `YYYY-mm-dd` format.

- **yearweek**: Year and week in `YYYYww` format.

- **product_number**: Unique ID for each product.

- **reporterhq_id**: Unique ID for each reseller (vendor).

- **prod_category**: Product category with a humorous or unconventional name, grouping products from the same product line.

- **specs**: Specifications of the product, which may include details like RAM, graphics card, or other components.

- **display_size**: Display or screen size of the PC.

- **segment**: Target segment of the product.

- **sales_units**: Sales to the final customer of the product for that week.

- **inventory_units**: Target variable representing inventory for each `product_number` and `reporterhq_id` for a specific week.

### Evaluation

Root Mean Squared Error (RMSE) is a popular evaluation metric used in many Kaggle competitions, especially in regression problems. It is a measure of the differences between predicted values and the actual values.

---

## ⚙️ Requirements

The `requirements.txt` file lists necessary libraries, simplifying setup. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Use

To utilize the **Forecaster** project, you'll need to import the relevant classes from the `utils` folder. Each class is designed to handle specific tasks within the time series forecasting pipeline.

### Importing classes

You can import the necessary classes as follows:

```python
from utils.data_preparation import DataPreparation
from utils.feature_engineering import FeatureEngineering
from utils.create_baselines import CreateBaselines
from utils.forecaster import Forecaster
from utils.evaluator import Evaluator
from utils.auxiliar import *
```

---

## Definitions

- **Cutoff**: The date from which the test set begins. It is crucial for splitting the dataset into training and testing subsets. The training data consists of all observations prior to the cutoff date, while the test set includes observations on or after this date. Properly defining the cutoff helps in evaluating the model's performance on unseen data.

---

This project is currently in development. If you have suggestions, feedback, or questions, please feel free to reach out!