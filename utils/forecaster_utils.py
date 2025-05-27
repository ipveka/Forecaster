#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Utility functions for the Forecaster package.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import logging

# Set plotting style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("talk")

def generate_sample_data(freq='W', periods=156):
    """
    Generate sample time series data with the specified frequency.
    
    Parameters:
    -----------
    freq : str, default='W'
        Frequency of the data ('D' for daily, 'W' for weekly, 'M' for monthly)
    periods : int, default=104
        Number of periods to generate
        
    Returns:
    --------
    pd.DataFrame
        Sample time series data
    """
    logging.info(f"Generating sample {freq} data with {periods} periods")
    
    # Set start date based on frequency (3 years back from 2022)
    if freq == 'D':
        start_date = datetime(2019, 1, 1)
    elif freq == 'W':
        start_date = datetime(2019, 1, 6)
    else:  # Monthly
        start_date = datetime(2019, 1, 31)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create product IDs and store IDs
    product_ids = ['P001', 'P002', 'P003', 'P004', 'P005']
    store_ids = ['S01', 'S02']
    
    # Create all combinations of dates, products, and stores
    records = []
    for date in dates:
        for product_id in product_ids:
            for store_id in store_ids:
                # Generate sample data with patterns
                day_of_year = date.timetuple().tm_yday
                
                # Base seasonal component (yearly seasonality)
                seasonal_component = np.sin(day_of_year / 365 * 2 * np.pi) * 20 + 50
                
                # Add product-specific trend
                if product_id == 'P001':
                    trend = 0.1 * (date - start_date).days / 30 
                elif product_id == 'P002':
                    trend = -0.05 * (date - start_date).days / 30 
                else:
                    trend = 0  # Flat trend
                
                # Add store-specific multiplier
                store_factor = 1.2 if store_id == 'S01' else 0.8
                
                # Add frequency-specific patterns
                if freq == 'D':
                    # Higher sales on weekends for daily data
                    weekday_factor = 1.3 if date.weekday() >= 5 else 1.0
                    noise_level = 5
                elif freq == 'W':
                    # No weekday factor for weekly data
                    weekday_factor = 1.0
                    noise_level = 3
                else:  # Monthly
                    # Higher sales in December for monthly data
                    weekday_factor = 1.5 if date.month == 12 else 1.0
                    noise_level = 2
                
                # Calculate final sales and inventory values
                base_value = seasonal_component + trend
                sales = float(base_value * weekday_factor * store_factor + np.random.normal(0, noise_level))
                inventory = float(sales * 2 + np.random.normal(0, noise_level))
                
                # Create a record
                record = {
                    'date': date,
                    'product': product_id,
                    'store': store_id,
                    'sales': max(0, sales),
                    'inventory': max(0, inventory)
                }
                records.append(record)
    
    # Create DataFrame and sort by date
    df = pd.DataFrame(records)
    df = df.sort_values(['date', 'product', 'store']).reset_index(drop=True)
    
    logging.info(f"Generated {len(df)} records")
    logging.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    return df


def visualize_data(df, freq, outputs_dir, target_col='sales'):
    """
    Create visualizations of the data and save to outputs folder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Data to visualize
    freq : str
        Frequency of the data
    outputs_dir : str
        Directory to save visualizations
    target_col : str, default='sales'
        Target column to visualize
    """
    logging.info("Creating initial data visualizations...")
    
    # Create a figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # Plot 1: Time series by product (averaged across stores)
    product_avg = df.groupby(['date', 'product'])[target_col].mean().reset_index()
    for product, data in product_avg.groupby('product'):
        axes[0].plot(data['date'], data[target_col], label=product)
    
    axes[0].set_title(f'{target_col.capitalize()} by Product ({freq} frequency)')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel(target_col.capitalize())
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Time series by store (averaged across products)
    store_avg = df.groupby(['date', 'store'])[target_col].mean().reset_index()
    for store, data in store_avg.groupby('store'):
        axes[1].plot(data['date'], data[target_col], label=store)
    
    axes[1].set_title(f'{target_col.capitalize()} by Store ({freq} frequency)')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel(target_col.capitalize())
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    
    # Save the visualization
    fig_path = os.path.join(outputs_dir, 'data_visualization.png')
    plt.savefig(fig_path)
    logging.info(f"Saved visualization to {fig_path}")
    plt.close()


def visualize_forecasts_by_cutoff(df, outputs_dir, target_col='sales', pred_col='prediction'):
    """
    Create visualizations of the forecasts with one plot per cutoff and save to outputs folder.
    Also creates a combined visualization with all cutoffs.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with actual values and predictions
    outputs_dir : str
        Directory to save visualizations
    target_col : str, default='sales'
        Target column with actual values
    pred_col : str, default='prediction'
        Column with predicted values
    """
    logging.info("Creating forecast visualizations with cutoff indicators...")
    
    # Find the prediction column if it's not the default
    if pred_col not in df.columns:
        pred_cols = [col for col in df.columns if 'pred' in col.lower()]
        if pred_cols:
            pred_col = pred_cols[0]
            logging.info(f"Using prediction column: {pred_col}")
        else:
            logging.warning("No prediction column found")
            return
    
    # Get unique cutoffs
    cutoffs = df['cutoff'].unique()
    logging.info(f"Creating forecast visualizations for {len(cutoffs)} cutoffs")
    
    # Create a combined visualization with one subplot per cutoff
    if len(cutoffs) > 0:
        # Create a single figure with subplots (one per cutoff)
        fig, axes = plt.subplots(len(cutoffs), 1, figsize=(15, 7 * len(cutoffs)))
        
        # If there's only one cutoff, make axes iterable
        if len(cutoffs) == 1:
            axes = [axes]
        
        # Look for baseline column if it exists
        baseline_cols = [col for col in df.columns if col.startswith(f'baseline_{target_col}_ma_')]
        baseline_col = baseline_cols[0] if baseline_cols else None
        
        # Plot each cutoff in its own subplot
        for i, cutoff_date in enumerate(cutoffs):
            ax = axes[i]
            cutoff_df = df[df['cutoff'] == cutoff_date].copy()
            cutoff_date_str = pd.Timestamp(cutoff_date).strftime('%Y-%m-%d')
            
            # Make sure sample column exists
            if 'sample' not in cutoff_df.columns:
                cutoff_timestamp = pd.Timestamp(cutoff_date)
                cutoff_df['sample'] = np.where(cutoff_df['date'] <= cutoff_timestamp, 'train', 'test')
            
            # Group by date and sample to get sum by train/test split
            data_by_sample = cutoff_df.groupby(['date', 'sample'])[[target_col, pred_col]].sum().reset_index()
            if baseline_col and baseline_col in cutoff_df.columns:
                # Include baseline in the groupby
                baseline_data = cutoff_df.groupby(['date', 'sample'])[baseline_col].sum().reset_index()
                # Merge baseline data with main data
                data_by_sample = pd.merge(data_by_sample, baseline_data, on=['date', 'sample'], how='left')
            
            # Get train and test data using the sample column
            train_data = data_by_sample[data_by_sample['sample'] == 'train']
            test_data = data_by_sample[data_by_sample['sample'] == 'test']
            
            # Plot train data
            if not train_data.empty:
                ax.plot(train_data['date'], train_data[target_col], 'b-', 
                        label=f'Actual (Train)', linewidth=2)
            
            # Plot test data and predictions
            if not test_data.empty:
                ax.plot(test_data['date'], test_data[target_col], 'g-', 
                        label=f'Actual (Test)', linewidth=2)
                ax.plot(test_data['date'], test_data[pred_col], 'r--', 
                        label=f'Predicted', linewidth=2)
                
                # Plot baseline if available
                if baseline_col and baseline_col in data_by_sample.columns:
                    ax.plot(test_data['date'], test_data[baseline_col], 'y--', 
                            label=f'Baseline (MA)', linewidth=1.5, alpha=0.7)
            
            # Add vertical line for cutoff date
            ax.axvline(x=cutoff_date, color='purple', linestyle='--', linewidth=2, 
                      label=f'Cutoff: {cutoff_date_str}')
            
            # Plot settings
            ax.set_title(f'Forecast Summary - Cutoff: {cutoff_date_str}')
            ax.set_xlabel('Date')
            ax.set_ylabel(target_col.capitalize())
            ax.legend()
            ax.grid(True)
        
        # Overall layout
        plt.tight_layout()
        
        # Save the combined visualization
        fig_path = os.path.join(outputs_dir, 'forecast_all_cutoffs.png')
        plt.savefig(fig_path)
        logging.info(f"Saved combined cutoff visualization to {fig_path}")
        plt.close()


def save_results(df, metrics, freq, outputs_dir):
    """
    Save the results to the outputs folder.
    
    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame with all results
    metrics : dict
        Dictionary of metrics
    freq : str
        Frequency of the data
    outputs_dir : str
        Directory to save results
    """
    logging.info("Saving results...")
    
    # Save the full DataFrame
    df_path = os.path.join(outputs_dir, 'forecaster_results.csv')
    df.to_csv(df_path, index=False)
    logging.info(f"Saved full results to {df_path}")
    
    # Save the metrics as a CSV
    if metrics:
        metrics_df = pd.DataFrame.from_dict({
            (model, metric): [value] 
            for model, model_metrics in metrics.items() 
            for metric, value in model_metrics.items()
        }, orient='columns')
        
        metrics_path = os.path.join(outputs_dir, 'forecaster_metrics.csv')
        metrics_df.to_csv(metrics_path, index=False)
        logging.info(f"Saved metrics to {metrics_path}")
        
        # Also save metrics by cutoff if available
        cutoffs = df['cutoff'].unique()
        if len(cutoffs) > 1:
            metrics_by_cutoff = {}
            for cutoff in cutoffs:
                cutoff_df = df[df['cutoff'] == cutoff]
                cutoff_metrics = {
                    'cutoff': str(cutoff),
                    'n_rows': len(cutoff_df),
                    'test_rows': len(cutoff_df[cutoff_df['sample'] == 'test']),
                    'train_rows': len(cutoff_df[cutoff_df['sample'] == 'train']),
                }
                metrics_by_cutoff[str(cutoff)] = cutoff_metrics
            
            cutoff_metrics_df = pd.DataFrame.from_dict(metrics_by_cutoff, orient='index')
            cutoff_metrics_path = os.path.join(outputs_dir, 'cutoff_stats.csv')
            cutoff_metrics_df.to_csv(cutoff_metrics_path)
            logging.info(f"Saved cutoff statistics to {cutoff_metrics_path}")


def get_frequency_params(freq):
    """
    Get the appropriate parameters for a given frequency.
    
    Parameters:
    -----------
    freq : str
        Frequency of the data ('D' for daily, 'W' for weekly, 'M' for monthly)
        
    Returns:
    --------
    dict
        Dictionary of parameters for the given frequency
    """
    if freq == 'D':
        # Daily 
        return {
            'horizon': 28,
            'dp_window_size': 14,
            'fe_window_size': (14, 28),
            'bs_window_size': 14,
            'lags': (7, 14, 28, 35),
            'periods': 1095  # 3 years of daily data
        }
    elif freq == 'W':
        # Weekly
        return {
            'horizon': 13,
            'dp_window_size': 13,
            'fe_window_size': (4, 13),
            'bs_window_size': 13,
            'lags': (13, 26, 39, 52),
            'periods': 156  # 3 years of weekly data
        }
    else:  
        # Monthly
        return {
            'horizon': 4,
            'dp_window_size': 4,
            'fe_window_size': (2, 6),
            'bs_window_size': 4,
            'lags': (4, 8, 12),
            'periods': 36  # 3 years of monthly data
        }
