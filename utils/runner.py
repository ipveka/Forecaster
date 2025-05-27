# General libraries
import pandas as pd
import numpy as np
import warnings
import os
import sys
import gc
import logging
import time
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import List, Dict, Union, Optional, Tuple, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Custom utilities
from utils.data_preparation import DataPreparation 
from utils.feature_engineering import FeatureEngineering
from utils.create_baselines import CreateBaselines 
from utils.forecaster import Forecaster
from utils.evaluator import Evaluator

# Suppress warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Set pandas display options for better output
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 1000)
pd.options.mode.chained_assignment = None

class Runner:
    """
    Runner class to orchestrate the entire Forecaster pipeline.
    
    This class provides a streamlined interface to run the complete forecasting process
    from data preparation to evaluation, by calling the wrapper functions from each
    component class in sequence.
    """
    
    def __init__(self, verbose=True):
        """
        Initialize the Runner class.
        
        Parameters:
        -----------
        verbose : bool, default=True
            Whether to print detailed information during execution.
        """
        self.metrics = None
        self.metric_table = None
        self.final_df = None
        self.verbose = verbose
        self.grouped_metrics = None
        self.execution_time = None
        self.feature_importances = None
        self.forecaster = None
        
        logging.info("Runner initialized successfully")
        if self.verbose:
            print("Runner initialized and ready to process forecasting pipeline.")
    
    def run_pipeline(self,
                    # Required parameters
                    df,
                    date_col, 
                    group_cols, 
                    signal_cols,
                    
                    # Data preparation parameters
                    target=None,  # Target column (if None, uses the first signal column)
                    horizon=4,    # Forecasting horizon
                    freq=None,    # Frequency of the data ('D', 'W', 'M')
                    n_cutoffs=1,  # Number of cutoffs for backtesting
                    complete_dataframe=True,  # Whether to fill in missing dates
                    smoothing=True,   # Whether to apply smoothing
                    dp_window_size=13,  # Window size for smoothing
                    
                    # Feature engineering parameters
                    fe_window_size=(4, 13),  # Window sizes for feature engineering
                    lags=(4, 13),         # Lag values for creating lag features
                    fill_lags=False,      # Whether to fill forward lags
                    n_clusters=10,        # Number of groups for quantile clustering
                    
                    # Baseline parameters
                    baseline_types=['MA', 'LR', 'ML'],  # Types of baselines to create
                    bs_window_size=13,                     # Window size for moving average baseline
                    
                    # Forecaster parameters
                    training_group=None,       # Column to use for training groups
                    model='LGBM',              # Model to use for forecasting
                    tune_hyperparameters=False,  # Whether to tune hyperparameters
                    search_method='halving',     # Search method for hyperparameter tuning
                    scoring='neg_mean_squared_log_error',  # Scoring metric for hyperparameter tuning
                    n_best_features=15,         # Number of best features to select
                    use_feature_selection=True,  # Whether to use feature selection
                    use_lags=True,              # Whether to include lag features
                    remove_outliers=False,       # Whether to remove outliers in forecasting
                    outlier_column=None,         # Column from which to remove outliers
                    lower_quantile=0.025,        # Lower quantile for outlier removal
                    upper_quantile=0.975,        # Upper quantile for outlier removal
                    ts_decomposition=False,      # Whether to use time series decomposition
                    baseline_col=None,           # Baseline column to use for comparison
                    use_guardrail=False,         # Whether to use guardrail limits
                    guardrail_limit=2.5,         # Limit value for guardrail adjustments
                    use_weights=False,           # Whether to use weights in training
                    use_parallel=True,           # Whether to run predictions in parallel
                    num_cpus=None,               # Number of CPUs to use for parallel processing
                    
                    # Evaluator parameters
                    eval_group_col=None,         # Column to group by for evaluation
                    eval_group_filter=None):     # Filter for groups in evaluation
        """
        Run the complete forecasting pipeline from data preparation to evaluation.
            
        Returns:
        --------
        pd.DataFrame
            The final DataFrame with all features, baselines, and predictions.
        """
        start_time = time.time()
        print("\n" + "=" * 80)
        print("STARTING FORECASTER PIPELINE")
        print("=" * 80)
        print(f"\nPipeline configuration:")
        print(f"• Date column: '{date_col}'")
        print(f"• Group columns: {group_cols}")
        print(f"• Signal columns: {signal_cols}")
        print(f"• Model: {model}")
        print(f"• Frequency: {freq or 'auto-detect'}")
        print(f"• Baseline types: {baseline_types}")
        logging.info(f"Starting pipeline with {len(df)} rows and {len(signal_cols)} signal columns")
        
        # Step 1: Data Preparation
        print("\n" + "-" * 80)
        print("[Step 1/5] RUNNING DATA PREPARATION")
        print("-" * 80)
        step_start = time.time()
        logging.info("Starting data preparation step")
        dp = DataPreparation()
        
        # If target is not specified, use the first signal column
        if target is None:
            target = signal_cols[0]
            print(f"Using '{target}' as the target column (first signal column)")
        elif target not in signal_cols:
            print(f"Warning: Specified target '{target}' not in signal_cols. Adding it.")
            signal_cols.append(target)
        else:
            print(f"Using specified target column: '{target}'")
        
        # Log key parameters
        print(f"Forecasting horizon: {horizon} periods")
        print(f"Number of cutoffs for backtesting: {n_cutoffs}")
        
        # Call the data preparation wrapper function with all relevant parameters
        prepared_df = dp.run_data_preparation(
            df=df.copy(), 
            group_cols=group_cols,
            date_col=date_col, 
            target=target,
            horizon=horizon,
            freq=freq,
            complete_dataframe=complete_dataframe,
            smoothing=smoothing,
            dp_window_size=dp_window_size,
            n_cutoffs=n_cutoffs
        )
                    
        # Log completion of data preparation step
        step_time = time.time() - step_start
        print(f"\nData preparation completed in {step_time:.2f} seconds")
        print("\n" + "-" * 80)
        print("[Step 2/5] RUNNING FEATURE ENGINEERING")
        print("-" * 80)
        step_start = time.time()
        logging.info("Starting feature engineering step")
        fe = FeatureEngineering()
        
        # Log feature engineering parameters
        print(f"Feature engineering configuration:")
        print(f"• Target column: '{target}'")
        print(f"• FE Window size: {fe_window_size}")
        print(f"• Lag periods: {lags}")
        print(f"• Fill lags: {fill_lags}")
        print(f"• Frequency: {freq}")
        
        # Call the feature engineering wrapper function with all relevant parameters
        feature_df = fe.run_feature_engineering(
            df=prepared_df,
            group_cols=group_cols,
            date_col=date_col,
            target=target,
            freq=freq,
            fe_window_size=fe_window_size,
            lags=lags,
            fill_lags=fill_lags,
            n_clusters=n_clusters
        )
        
        # Log completion of feature engineering step
        step_time = time.time() - step_start
        print(f"\nFeature engineering completed in {step_time:.2f} seconds")
        print("\n" + "-" * 80)
        print("[Step 3/5] CREATING BASELINES")
        print("-" * 80)
        step_start = time.time()
        logging.info(f"Starting baseline creation with types: {baseline_types}")
        cb = CreateBaselines()
        
        # Log baseline parameters
        print(f"Baseline configuration:")
        print(f"• Baseline types: {baseline_types}")
        print(f"• Window size: {bs_window_size}")
        
        # Call the baseline creation wrapper function with all relevant parameters
        baseline_df = cb.run_baselines(
            df=feature_df,
            group_cols=group_cols,
            date_col=date_col,
            signal_cols=[target],
            baseline_types=baseline_types,
            bs_window_size=bs_window_size
        )
        
        # Log completion of baseline creation step
        step_time = time.time() - step_start
        print(f"\nBaseline creation completed in {step_time:.2f} seconds")
        print("\n" + "-" * 80)
        print("[Step 4/5] RUNNING FORECASTING")
        print("-" * 80)
        step_start = time.time()
        logging.info(f"Starting forecasting with model: {model}")
        forecaster = Forecaster(baseline_df)
        
        # Log forecasting parameters
        print("\nPreparing dataframe for forecasting...")
        
        # If training_group is not specified, create a default one
        if training_group is None or training_group not in baseline_df.columns:
            print(f"Creating default training_group with value 1")
            baseline_df['training_group'] = 1
            training_group = 'training_group'
            
        # Ensure the forecaster has the updated dataframe with the training_group column
        forecaster = Forecaster(baseline_df)
        
        # Set baseline column if not specified
        if baseline_col is None:
            baseline_col = f"baseline_{target}_ma_{bs_window_size}"
        
        # Log forecasting configuration
        print(f"Forecasting configuration:")
        print(f"• Model: {model}")
        print(f"• Target column: '{target}'")
        print(f"• Training group column: '{training_group}'")
        print(f"• Tune hyperparameters: {tune_hyperparameters}")
        print(f"• Use feature selection: {use_feature_selection}")
        print(f"• Use lag features: {use_lags}")
        print(f"• Use guardrail: {use_guardrail}")
        print(f"• Use parallel processing: {use_parallel}")
        print(f"• Remove outliers: {remove_outliers}")
        
        # Automatically find all feature columns containing 'feature' in their names
        feature_cols = [col for col in baseline_df.columns if "feature" in col]
        
        # Filter out lag features if use_lags is False
        if not use_lags:
            original_count = len(feature_cols)
            feature_cols = [col for col in feature_cols if "lag" not in col]
            filtered_count = original_count - len(feature_cols)
            print(f"Found {original_count} feature columns, filtered out {filtered_count} lag features")
        else:
            print(f"Found {len(feature_cols)} feature columns for forecasting")
        
        # Print all features that will be used
        print("\nFeatures to be used in forecasting:")
        for i, feature in enumerate(feature_cols, 1):
            print(f"{i}. {feature}")
        print()
        
        # Call the forecasting function with all relevant parameters
        forecast_df = forecaster.run_backtesting(
            group_cols=group_cols,
            features=feature_cols,
            params=None,
            training_group=training_group,
            target_col=target,
            model=model,
            tune_hyperparameters=tune_hyperparameters,
            search_method=search_method,
            param_distributions=None,
            scoring=scoring,
            n_iter=50,
            best_features=None, 
            n_best_features=n_best_features,
            remove_outliers=False,
            outlier_column=None,
            lower_quantile=0.025,
            upper_quantile=0.975,
            ts_decomposition=False,
            baseline_col=baseline_col,
            use_guardrail=use_guardrail,
            guardrail_limit=guardrail_limit,
            use_weights=use_weights,
            use_parallel=use_parallel,
            num_cpus=num_cpus
        )
        
        # Log completion of forecasting step
        step_time = time.time() - step_start
        print(f"\nForecasting completed in {step_time:.2f} seconds")
        print("\n" + "-" * 80)
        print("[Step 5/5] EVALUATING RESULTS")
        print("-" * 80)
        step_start = time.time()
        logging.info("Starting evaluation step")
        
        # Log evaluation parameters
        print(f"Evaluation configuration:")
        print(f"• Target column: '{target}'")
        print(f"• Baseline column: '{baseline_col}'")
        print(f"• Prediction columns: ['prediction']")
        
        # Store the final dataframe
        self.final_df = forecast_df
        self.execution_time = time.time() - start_time
        
        # Store the forecaster instance for later use (e.g., feature importance)
        self.forecaster = forecaster
        
        # Create an evaluator
        evaluator = Evaluator(
            df=forecast_df,
            actuals_col=target,
            baseline_col=baseline_col,
            preds_cols=['prediction']
        )
        
        # Calculate metrics
        self.metrics = evaluator.evaluate()
        self.metric_table = evaluator.create_metric_table()
        
        # Print metric summary
        print("\nMetric summary:\n")
        print(self.metric_table)
        
        # If a grouping column is specified, calculate grouped metrics
        if eval_group_col and eval_group_col in forecast_df.columns:
            self.grouped_metrics = evaluator.calculate_grouped_metric(
                metric_name='RMSE',
                group_col=eval_group_col,
                group_filter=eval_group_filter
            )
            
            # Print grouped metrics
            print(f"\nRMSE by {eval_group_col}:")
            print(self.grouped_metrics)
        
        # Store feature importances if available
        if hasattr(forecaster, 'feature_importances') and forecaster.feature_importances:
            self.feature_importances = forecaster.get_feature_importance()
            print("\nFeature importance information is available. Use plot_feature_importance() to visualize.")
        
        # Final message and return
        print("\n" + "=" * 80)
        print("FORECASTER PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"\nTotal execution time: {self.execution_time:.2f} seconds")
        print(f"Data shape: {forecast_df.shape[0]} rows × {forecast_df.shape[1]} columns")
        
        if self.metric_table is not None:
            print(f"\nMetrics summary:")
            print(self.metric_table)
            
        logging.info(f"Pipeline completed successfully in {self.execution_time:.2f} seconds")
        
        # Free memory
        gc.collect()
        
        return forecast_df
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8), save_path=None):
        """
        Plot the feature importance from the forecaster.
        
        Parameters:
        -----------
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 8)
            Figure size
        save_path : str, default=None
            Path to save the figure. If None, the figure is displayed but not saved.
            
        Returns:
        --------
        None
        """
        if self.forecaster is None:
            print("No forecaster available. Run the pipeline first.")
            return
        
        if not hasattr(self.forecaster, 'feature_importances') or not self.forecaster.feature_importances:
            print("No feature importance information available.")
            return
        
        # Get feature importances
        all_importances = []
        all_feature_names = []
        
        # Collect all model feature importances and names
        for (cutoff, group), model in self.forecaster.models.items():
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = model.feature_name_
                
                if len(importances) > 0:
                    all_importances.append(importances)
                    all_feature_names.append(feature_names)
        
        if not all_importances:
            print("No feature importance data found in models.")
            return
            
        # Use the feature names from the first model
        feature_names = all_feature_names[0]
        
        # Average feature importances across all models
        avg_importances = np.mean(all_importances, axis=0)
        
        # Create a dictionary of feature name to importance
        importance_dict = {}
        for i, name in enumerate(feature_names):
            if i < len(avg_importances):
                importance_dict[name] = avg_importances[i]
        
        # Sort features by importance and get top N
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
        features = [item[0] for item in sorted_features]
        importance_values = [item[1] for item in sorted_features]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(features, importance_values, color='steelblue')
        plt.xlabel('Average Importance', fontsize=14)
        plt.ylabel('Features', fontsize=14)
        plt.title('Top Feature Importance Across All Models', fontsize=16)
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        
        # Save if path is provided
        if save_path:
            plt.savefig(save_path)
            print(f"Feature importance plot saved to: {save_path}")
    
    def get_metrics(self):
        """
        Get the evaluation metrics dictionary.
        
        Returns:
        --------
        dict
            Dictionary containing metrics for baseline and predictions.
        """
        return self.metrics
    
    def get_metric_table(self):
        """
        Get the formatted metric table.
        
        Returns:
        --------
        pd.DataFrame
            DataFrame containing the rounded metrics for baseline and predictions.
        """
        return self.metric_table
    
    def get_final_df(self):
        """
        Get the final DataFrame with all features, baselines, and predictions.
        
        Returns:
        --------
        pd.DataFrame
            The final DataFrame.
        """
        return self.final_df
    
    def get_execution_time(self):
        """
        Get the total execution time of the pipeline in seconds.
        
        Returns:
        --------
        float or None
            Total execution time in seconds, or None if pipeline hasn't been run.
        """
        return self.execution_time
    
    def get_grouped_metrics(self):
        """
        Get the grouped metrics if available.
        
        Returns:
        --------
        pd.DataFrame or None
            DataFrame containing metrics grouped by a specified column,
            or None if grouped metrics were not calculated.
        """
        return self.grouped_metrics
    
    def get_feature_importances(self):
        """
        Get the feature importances if available.
        
        Returns:
        --------
        dict or None
            Dictionary containing feature importances for each cutoff and training group,
            or None if feature importances were not calculated.
        """
        if self.forecaster:
            return self.forecaster.get_feature_importance()
        return None
