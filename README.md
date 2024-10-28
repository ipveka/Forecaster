# ⚡ LightGBM-Forecaster

**LightGBM-Forecaster** is a time series forecasting project using **LightGBM**. It streamlines the forecasting pipeline with dedicated classes for each stage, from data preparation to model evaluation, making it adaptable to various time series forecasting use cases.

Created initially for the [Forecasting competition VN1 Forecasting](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description), this project can be tailored to other datasets and forecasting needs.

You can download the data from: https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn1-forecasting-accuracy-challenge-phase-1/datasets

## ⚙️ Why LightGBM is Great for Forecasting

LightGBM, or **Light Gradient Boosting Machine**, is a high-performance algorithm designed for fast and efficient model building, making it ideal for complex forecasting tasks. Here’s a quick breakdown of how it works and why it’s so effective for time series forecasting:

### 🌿 **How LightGBM Works**

LightGBM builds prediction models by iteratively creating trees that improve on previous ones. Key aspects include:

- **Leaf-Wise Tree Growth**: Optimizes trees by focusing on the branches with the highest error reduction, leading to more accurate models.
- **Categorical Feature Handling**: Handles categorical data like days or seasons directly, without extra preprocessing.
- **Histogram-Based Learning**: Groups continuous values into discrete bins, reducing computation and speeding up training.

### ⏱ **Why It’s Efficient for Time Series Forecasting**

1. **Scalable for Large Datasets**: Ideal for handling complex, high-dimensional time series data, especially when many features are generated.
2. **Captures Patterns and Trends**: Tree-based structure can capture nonlinear relationships and seasonal patterns.
3. **Good for Grouped Data**: Supports segmented models for multiple groups, useful in multi-series forecasts like client or product-level forecasting.
4. **Handles Missing and Noisy Data**: Naturally deals with missing values and is robust to outliers, both common in real-world data.

---

## 🌟 Highlights

The `LightGBM-Forecaster` project offers several classes that help create features to boost accuracy and efficiency in forecasting:

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
   Additionally, a function named `run_data_preparation` orchestrates all these steps, allowing users to perform comprehensive data preparation in one go.

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
   The `run_feature_engineering` function enables users to orchestrate all feature engineering steps in a single operation.

- **`CreateBaselines`** 🧩  
   This class uses the outputs from `FeatureEngineering` to establish baseline models that serve as benchmarks, enabling users to evaluate LightGBM’s improvements over simpler models. It also generates simple forecasts for numeric signals.

- **`Forecaster`** 🔮  
   The core class for training the LightGBM model, `Forecaster` efficiently manages the forecasting process. This class leverages the features generated by `FeatureEngineering` and `CreateBaselines` to produce forecasts at each cutoff. Key capabilities include:
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

  You can download the data from the examples here: https://www.datasource.ai/en/home/data-science-competitions-for-startups/vn1-forecasting-accuracy-challenge-phase-1/datasets

- **`docs`**  
  Comprehensive project documentation, detailing the classes, functions, architecture, and usage guidelines.

- **`notebooks`**  
  Jupyter notebooks demonstrating the LightGBM-Forecaster workflow, including:
  - **`data_preparation`**: Walks through each step of data preparation, feature engineering, and baseline creation individually, allowing users to customize executions.
  - **`lightgbm`**: Covers model evaluation, hyperparameter tuning, feature selection, and forecasting using `Forecaster` class, with insights on both training and validation processes.
  - **`submission`**: Prepares the final forecast output, formats it for submission, and includes post-processing steps to ensure compatibility with the desired output structure.
  - **`runner`**: An end-to-end workflow notebook utilizing the `run_data_preparation` and `run_feature_engineering` orchestrators to streamline the entire process, combined with `Forecaster` and `Evaluator` calls to create a complete forecasting pipeline.

- **`submissions`**  
  Contains the forecast model outputs, including predictions and performance metrics.

- **`utils`**  
  The utility folder holding the main classes, auxiliary functions, and plotting tools for the project.

---

### 📈 **Notebooks**  

Following the use case for the Forecasting competition VN1, this project contains end to end notebook examples using the data provided for the competition. The dataset used for both phases look like this:

- **Sales Data** (Phase X - `Sales.csv`): Weekly sales units.
- **Price Data** (Phase X - `Price.csv`): Pricing data based on actual transactions.

In the data preparation notebook we join the tables from all phases. 

All information is provided at Client-Warehouse-Product-Week level

---

## Definitions

- **Cutoff**: The date from which the test set begins. It is crucial for splitting the dataset into training and testing subsets. The training data consists of all observations prior to the cutoff date, while the test set includes observations on or after this date. Properly defining the cutoff helps in evaluating the model's performance on unseen data.

---

## ⚙️ Requirements

The `requirements.txt` file lists necessary libraries, simplifying setup. Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Use

To utilize the **LightGBM-Forecaster** project, you'll need to import the relevant classes from the `utils` folder. Each class is designed to handle specific tasks within the time series forecasting pipeline.

---

## Importing classes

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

This project is currently in development. If you have suggestions, feedback, or questions, please feel free to reach out!