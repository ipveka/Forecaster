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

def generate_sample_data(freq='W', periods=104):
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
    
    # Set start date based on frequency
    if freq == 'D':
        start_date = datetime(2022, 1, 1)
    elif freq == 'W':
        start_date = datetime(2022, 1, 3)  # Sunday
    else:  # Monthly
        start_date = datetime(2022, 1, 31)
    
    # Create date range
    dates = pd.date_range(start=start_date, periods=periods, freq=freq)
    
    # Create product IDs and store IDs
    product_ids = ['P001', 'P002', 'P003']
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
                    trend = 0.1 * (date - start_date).days / 30  # Increasing trend
                elif product_id == 'P002':
                    trend = -0.05 * (date - start_date).days / 30  # Decreasing trend
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
                    'sales': max(0, sales),  # Ensure non-negative
                    'inventory': max(0, inventory)  # Ensure non-negative
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
    fig_path = os.path.join(outputs_dir, f'data_visualization_{freq}.png')
    plt.savefig(fig_path)
    logging.info(f"Saved visualization to {fig_path}")
    plt.close()


def visualize_forecasts_by_cutoff(df, outputs_dir, target_col='sales', pred_col='prediction'):
    """
    Create visualizations of the forecasts with one plot per cutoff and save to outputs folder.
    
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
    
    # Create one visualization per cutoff
    for cutoff_date in cutoffs:
        cutoff_df = df[df['cutoff'] == cutoff_date].copy()
        
        # Create a figure with three subplots
        fig, axes = plt.subplots(3, 1, figsize=(15, 18))
        cutoff_date_str = pd.Timestamp(cutoff_date).strftime('%Y-%m-%d')
        
        # Overall plot
        overall_avg = cutoff_df.groupby('date')[[target_col, pred_col]].mean().reset_index()
        
        # Split train/test data for better visualization
        train_data = overall_avg[cutoff_df['sample'].iloc[0] == 'train']
        test_data = overall_avg[cutoff_df['sample'].iloc[0] == 'test']
        
        # Plot train data
        if not train_data.empty:
            axes[0].plot(train_data['date'], train_data[target_col], 'b-', 
                        label=f'Actual (Train)', linewidth=2)
        
        # Plot test data
        if not test_data.empty:
            axes[0].plot(test_data['date'], test_data[target_col], 'g-', 
                        label=f'Actual (Test)', linewidth=2)
            axes[0].plot(test_data['date'], test_data[pred_col], 'r--', 
                        label=f'Predicted', linewidth=2)
        
        # Add vertical line for cutoff date
        axes[0].axvline(x=cutoff_date, color='purple', linestyle='--', linewidth=2, 
                      label=f'Cutoff: {cutoff_date_str}')
        
        axes[0].set_title(f'Overall Forecast - Cutoff: {cutoff_date_str}')
        axes[0].set_xlabel('Date')
        axes[0].set_ylabel(target_col.capitalize())
        axes[0].legend()
        axes[0].grid(True)
        
        # Plot by product
        # Get product data
        product_df = cutoff_df.groupby(['date', 'product', 'sample'])[[target_col, pred_col]].mean().reset_index()
        
        # Get unique products
        products = product_df['product'].unique()
        colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'orange']
        
        for i, product in enumerate(products):
            product_data = product_df[product_df['product'] == product]
            color = colors[i % len(colors)]
            
            # Split train/test data for this product
            train_product = product_data[product_data['sample'] == 'train']
            test_product = product_data[product_data['sample'] == 'test']
            
            # Plot train data
            if not train_product.empty:
                axes[1].plot(train_product['date'], train_product[target_col], 
                            color=color, linestyle='-',
                            label=f'{product} Actual (Train)')
            
            # Plot test data
            if not test_product.empty:
                axes[1].plot(test_product['date'], test_product[target_col], 
                            color=color, linestyle='-', linewidth=2,
                            label=f'{product} Actual (Test)')
                axes[1].plot(test_product['date'], test_product[pred_col], 
                            color=color, linestyle='--', linewidth=2,
                            label=f'{product} Predicted')
        
        # Add vertical line for cutoff date
        axes[1].axvline(x=cutoff_date, color='purple', linestyle='--', linewidth=2, 
                      label=f'Cutoff: {cutoff_date_str}')
        
        axes[1].set_title(f'Forecast by Product - Cutoff: {cutoff_date_str}')
        axes[1].set_xlabel('Date')
        axes[1].set_ylabel(target_col.capitalize())
        axes[1].legend()
        axes[1].grid(True)
        
        # Plot by store
        # Get store data
        store_df = cutoff_df.groupby(['date', 'store', 'sample'])[[target_col, pred_col]].mean().reset_index()
        
        # Get unique stores
        stores = store_df['store'].unique()
        
        for i, store in enumerate(stores):
            store_data = store_df[store_df['store'] == store]
            color = colors[i % len(colors)]
            
            # Split train/test data for this store
            train_store = store_data[store_data['sample'] == 'train']
            test_store = store_data[store_data['sample'] == 'test']
            
            # Plot train data
            if not train_store.empty:
                axes[2].plot(train_store['date'], train_store[target_col], 
                           color=color, linestyle='-',
                           label=f'{store} Actual (Train)')
            
            # Plot test data
            if not test_store.empty:
                axes[2].plot(test_store['date'], test_store[target_col], 
                           color=color, linestyle='-', linewidth=2,
                           label=f'{store} Actual (Test)')
                axes[2].plot(test_store['date'], test_store[pred_col], 
                           color=color, linestyle='--', linewidth=2,
                           label=f'{store} Predicted')
        
        # Add vertical line for cutoff date
        axes[2].axvline(x=cutoff_date, color='purple', linestyle='--', linewidth=2, 
                      label=f'Cutoff: {cutoff_date_str}')
        
        axes[2].set_title(f'Forecast by Store - Cutoff: {cutoff_date_str}')
        axes[2].set_xlabel('Date')
        axes[2].set_ylabel(target_col.capitalize())
        axes[2].legend()
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save the visualization
        fig_path = os.path.join(outputs_dir, f'forecast_cutoff_{cutoff_date_str}_{freq}.png')
        plt.savefig(fig_path)
        logging.info(f"Saved forecast visualization for cutoff {cutoff_date_str} to {fig_path}")
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
    df_path = os.path.join(outputs_dir, f'forecaster_results_{freq}.csv')
    df.to_csv(df_path, index=False)
    logging.info(f"Saved full results to {df_path}")
    
    # Save the metrics as a CSV
    if metrics:
        metrics_df = pd.DataFrame.from_dict({
            (model, metric): [value] 
            for model, model_metrics in metrics.items() 
            for metric, value in model_metrics.items()
        }, orient='columns')
        
        metrics_path = os.path.join(outputs_dir, f'forecaster_metrics_{freq}.csv')
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
            cutoff_metrics_path = os.path.join(outputs_dir, f'cutoff_stats_{freq}.csv')
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
        return {
            'horizon': 7,
            'window_size': 7,
            'ma_window_size': 7,
            'window_sizes': (3, 7),
            'lags': (1, 7),
            'periods': 365
        }
    elif freq == 'W':
        return {
            'horizon': 4,
            'window_size': 4,
            'ma_window_size': 4,
            'window_sizes': (2, 8),
            'lags': (1, 8),
            'periods': 104
        }
    else:  # Monthly
        return {
            'horizon': 3,
            'window_size': 3,
            'ma_window_size': 3,
            'window_sizes': (2, 12),
            'lags': (1, 12),
            'periods': 36
        }
