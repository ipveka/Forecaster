# ⚡ LightGBM-Forecaster 

**LightGBM-Forecaster** is a time series forecasting project powered by **LightGBM**. It streamlines the forecasting pipeline with dedicated classes for each stage, from data preparation to model evaluation, making it adaptable to various time series forecasting use cases.

Created initially for the [Forecasting competition VN1 Forecasting](https://www.datasource.ai/en/home/data-science-competitions-for-startups/phase-2-vn1-forecasting-accuracy-challenge/description), this project can be tailored to other datasets and forecasting needs.

---


## 🌟 Highlights

The `LightGBM-Forecaster` project offers several classes that help create features to boost accuracy and efficiency in forecasting:

- **Backtesting Capabilities** 🧭: Simulates real-world model performance by evaluating it over time.
  
- **Automatic Best Feature Selection** 🎯: Optimizes feature relevance automatically for enhanced model performance.
  
- **Outlier Detection and Treatment** 🚨: Identifies and reduces the impact of outliers for more robust forecasting.

- **Support for Grouped Data** 👥: Generates models for various data segments, tailoring forecasts to each.

- **Parallel and GPU Computing** 🚀: Accelerates training with parallel processing and GPU support.

---

## 🔍 **Classes**

The project is built around five primary classes, each handling specific stages in the forecasting pipeline:

- **`DataPreparation`** 🛠️  
   Prepares datasets by handling missing values, transforming data types, and ensuring a clean, model-ready structure.

- **`FeatureEngineering`** 🔍  
   Generates and refines relevant features, including time-based transformations, clustering and segmentation techniques to optimize input data for LightGBM.

- **`CreateBaselines`** 🧩  
   Develops baseline models as a benchmark, enabling users to compare LightGBM's improvements against simpler models.

- **`Forecaster`** 🔮  
   The core class, responsible for training the LightGBM model and managing the forecasting tasks with efficiency and precision.

- **`Evaluator`** 📊  
   Evaluates model performance with key metrics and visualizations, helping users interpret forecasting accuracy and reliability.

---

## 📁 Project Structure

The repository is organized into several folders for easy navigation:

- **`data`**  
  Contains both raw and processed datasets. The processed datasets are ready for model training following data cleaning and feature engineering.

- **`docs`**  
  Comprehensive project documentation, detailing the classes, functions, architecture, and usage guidelines.

- **`notebooks`** 📝  
  Interactive Jupyter notebooks demonstrating the LightGBM-Forecaster workflow, including:
  - **`data_preparation`**: Data prep, backtesting DataFrame creation, and baseline modeling.
  - **`lightgbm`**: Model evaluation and forecasting using LightGBM.
  - **`submission`**: Guides final output formatting for submission.
  - **`runner`**: An end-to-end workflow notebook (currently in development).

- **`submissions`**  
  Contains the forecast model outputs, including predictions and performance metrics.

- **`utils`** 🔧  
  The utility folder holding the main classes, auxiliary functions, and plotting tools for the project.

---

### 📈 **Notebooks**  

Following the use case for the Forecasting competition VN1, this project contains end to end notebook examples using the data provided for the competition. The dataset used for both phases look like this:

- **Sales Data** (Phase X - `Sales.csv`): Weekly sales units.

- **Price Data** (Phase X - `Price.csv`): Pricing data based on actual transactions.

In the data preparation notebook we join the tables from all phases. 

All information is provided at Client-Warehouse-Product-Week level

---

## ⚙️ Requirements

The `requirements.txt` file lists necessary libraries, simplifying setup. Install dependencies with:

```bash
pip install -r requirements.txt
```

## How to Use

To utilize the **LightGBM-Forecaster** project, you'll need to import the relevant classes from the `utils` folder. Each class is designed to handle specific tasks within the time series forecasting pipeline.

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