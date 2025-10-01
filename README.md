# ⚡ Forecaster

A production-ready time series forecasting framework using **LightGBM** and gradient boosting models. Built for scalability, it handles multi-series forecasting with automated feature engineering, backtesting, and model evaluation.

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Basic Usage

```python
from utils.runner import Runner

# Initialize runner
runner = Runner()

# Run complete pipeline
result_df = runner.run_pipeline(
    df=your_data,
    date_col="date",
    group_cols=["product", "store"],
    signal_cols=["sales"],
    target="sales",
    freq="W",  # W=weekly, D=daily, M=monthly
    horizon=12,
    n_cutoffs=3
)
```

### Command Line

```bash
python src/run_forecaster.py --freq W --horizon 12 --n_cutoffs 3 --model LGBM
```

---

## 🎯 Key Features

- **Multi-Series Forecasting**: Handle thousands of time series simultaneously
- **Automated Feature Engineering**: Lags, moving averages, seasonality, trends
- **Backtesting**: Multiple cutoffs for robust validation
- **Frequency-Aware**: Auto-detects and adapts to daily/weekly/monthly data
- **Model Selection**: LightGBM, Random Forest, GBM, AdaBoost, Linear Regression
- **Baseline Comparison**: MA, LR, and LGBM baselines for benchmarking

---

## 📊 Pipeline Components

### 1. Data Preparation (`DataPreparation`)
- Fills missing dates for complete time series
- Smooths target variables to reduce noise
- Creates train/test splits with multiple cutoffs
- Ensures valid target data for cutoff selection

### 2. Feature Engineering (`FeatureEngineering`)
- **Temporal**: Date features (day, week, month, year, cyclical encodings)
- **Lags**: Configurable lag periods with forward fill option
- **Rolling Stats**: Moving averages, min, max, std
- **Categorical**: Label encoding for grouping variables
- **Clusters**: Quantile and history-based clustering
- **Weights**: Time-based sample weights for recent data emphasis

### 3. Baseline Creation (`CreateBaselines`)
- Moving Average (MA) baselines
- Linear Regression (LR) baselines
- LightGBM (LGBM) baselines
- Feature and prediction baselines

### 4. Forecasting (`Forecaster`)
- Model training with hyperparameter tuning
- Feature selection for efficiency
- Outlier detection and treatment
- Training group support for segmented models
- Parallel processing capability

### 5. Evaluation (`Evaluator`)
- RMSE, MAE, MAPE, R² metrics
- Lag-specific accuracy analysis
- Visualization tools for forecast quality

---

## 📁 Project Structure

```
Forecaster/
├── data/              # Raw and processed datasets
├── docs/              # Detailed documentation per class
├── notebooks/         # Example workflows
│   ├── data_preparation.ipynb
│   ├── forecasting.ipynb
│   └── runner.ipynb
├── src/               # Executable scripts
│   ├── run_forecaster.py
│   └── generate_examples.py
├── utils/             # Core classes and utilities
│   ├── data_preparation.py
│   ├── feature_engineering.py
│   ├── create_baselines.py
│   ├── forecaster.py
│   ├── evaluator.py
│   ├── runner.py
│   └── forecaster_utils.py
└── outputs/           # Results and visualizations
```

---

## 🔧 Configuration

### Frequency-Based Parameters

The framework automatically selects optimal parameters based on data frequency:

| Parameter | Daily (D) | Weekly (W) | Monthly (M) |
|-----------|-----------|------------|-------------|
| Horizon | 30 | 13 | 6 |
| Lags | [1,2,3,7,14,30] | [1,2,4,13,26] | [1,2,3,6,12] |
| FE Window | [7,14,30] | [4,8,13] | [3,6,12] |
| DP Window | 7 | 4 | 3 |
| BS Window | 7 | 4 | 3 |

### Parameter Naming Convention

- **`dp_window_size`**: Data preparation smoothing window
- **`fe_window_size`**: Feature engineering rolling windows
- **`bs_window_size`**: Baseline calculation window

---

## 💡 Example Use Case

Based on the HP Supply Chain Optimization challenge (HackUPC 2023), this framework forecasts inventory levels for multiple products on a weekly basis.

**Data Source**: [Kaggle Competition](https://www.kaggle.com/competitions/hp-supply-chain-optimization)

**Key Fields**:
- `date`: Date of record
- `product_number`: Product ID
- `reporterhq_id`: Reseller ID
- `sales_units`: Sales to final customer
- `inventory_units`: Target variable (inventory level)

---

## 🔑 Key Concepts

### Cutoff
The date separating training and test data. The framework:
- Always includes the **latest date with valid target** as the primary cutoff (for forecasting)
- Adds historical cutoffs for backtesting validation
- Ensures cutoffs are based only on dates where `target > 0` and not NaN

### Valid Target Data
The framework filters data to ensure cutoffs are created only from dates where:
- Target value is greater than 0
- Target value is not NaN
- This prevents issues with forecast horizons extending beyond actual data

---

## 📈 Why LightGBM?

- **Speed**: Optimized for large datasets with millions of rows
- **Accuracy**: Leaf-wise tree growth captures complex patterns
- **Flexibility**: Handles categorical variables natively
- **Scalability**: GPU support and parallelization
- **Robustness**: Built-in handling of missing data and outliers

---

## 🎯 Available Models

- **LightGBM (LGBM)**: Default choice for speed and accuracy
- **Random Forest (RF)**: Ensemble method for robust predictions
- **Gradient Boosting (GBM)**: Sequential tree building
- **AdaBoost (ADA)**: Weak learner boosting
- **Linear Regression (LR)**: Simple baseline model

---

## 📚 Documentation

Detailed documentation for each component is available in the `docs/` folder:
- `DataPreparation.md`
- `FeatureEngineering.md`
- `CreateBaselines.md`
- `Forecaster.md`
- `Evaluator.md`
- `Runner.md`
- `ForecasterUtils.md`

---

## 🛠️ Development Status

This project is actively maintained. Contributions, feedback, and suggestions are welcome!
