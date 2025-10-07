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
- **Smart Baselines**: MA, Linear Regression, LightGBM, and **Croston** for intermittent demand
- **Smart Ensembles**: Intelligent combination of ML predictions and baselines
- **Backtesting**: Multiple cutoffs for robust validation
- **Frequency-Aware**: Auto-detects and adapts to daily/weekly/monthly data
- **Multiple Models**: LightGBM, Random Forest, GBM, AdaBoost, Linear Regression

---

## 📊 Pipeline

1. **Data Preparation** → Fill missing dates, smooth signals, create cutoffs
2. **Feature Engineering** → Temporal features, lags, rolling stats, clusters
3. **Baseline Creation** → MA, LR, LGBM, and Croston baselines
4. **Forecasting** → Train models with hyperparameter tuning
5. **Evaluation** → RMSE, MAE, MAPE metrics with smart ensemble

---

## 🔧 Configuration

The framework automatically selects optimal parameters based on data frequency:

| Parameter | Daily (D) | Weekly (W) | Monthly (M) |
|-----------|-----------|------------|-------------|
| Horizon | 30 | 13 | 6 |
| Lags | [1,2,3,7,14,30] | [1,2,4,13,26] | [1,2,3,6,12] |
| Windows | [7,14,30] | [4,8,13] | [3,6,12] |

---

## 🎯 Available Models

- **LightGBM (LGBM)**: Default choice for speed and accuracy
- **Random Forest (RF)**: Ensemble method for robust predictions
- **Gradient Boosting (GBM)**: Sequential tree building
- **AdaBoost (ADA)**: Weak learner boosting
- **Linear Regression (LR)**: Simple baseline model

---

## 📁 Project Structure

```
Forecaster/
├── data/              # Raw and processed datasets
├── notebooks/         # Example workflows
├── src/               # Executable scripts
├── utils/             # Core classes and utilities
├── api/               # REST API
└── outputs/           # Results and visualizations
```

---

## 💡 Example Use Case

Based on the HP Supply Chain Optimization challenge, this framework forecasts inventory levels for multiple products on a weekly basis.

**Key Fields**:
- `date`: Date of record
- `product_number`: Product ID
- `reporterhq_id`: Reseller ID
- `sales_units`: Sales to final customer
- `inventory_units`: Target variable (inventory level)

---

## 🛠️ Development Status

This project is actively maintained. Contributions, feedback, and suggestions are welcome!