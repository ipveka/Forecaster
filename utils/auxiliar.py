# General libraries
import pandas as pd
import numpy as np
import warnings
import psutil
import gc
import os

# Data preparation
from itertools import product

# Plots
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from IPython import display
from time import sleep
from math import ceil

# Sklearn
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# Sklearn
from sklearn.preprocessing import LabelEncoder

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Auxiliar functions

# Unpivot data
def unpivot_data(df, id_vars, var_name='Date', value_name='Price'):
    """
    Unpivots a given sales dataframe.

    Parameters:
    - df (DataFrame): The sales dataframe to unpivot.
    - id_vars (list): Columns to keep as identifier variables (e.g., ['Client', 'Warehouse', 'Product']).
    - var_name (str): The name of the new column that will hold the dates. Default is 'Date'.
    - value_name (str): The name of the new column that will hold the unpivoted values. Default is 'Price'.

    Returns:
    - DataFrame: The unpivoted dataframe with date and price columns.
    """
    # Unpivot the DataFrame
    df_unpiv = pd.melt(df,
                       id_vars=id_vars,
                       var_name=var_name,
                       value_name=value_name)

    # Convert the 'Date' column from string to datetime format
    df_unpiv[var_name] = pd.to_datetime(df_unpiv[var_name])

    return df_unpiv

# Function to create a single axis plot for a specific entity
def create_single_axis_plot(ax, df, entity, group_col, cutoff, baseline_col, target_col, title):
    """
    Create a single-axis plot for the specified entity on the given axis.

    :param ax: The axis to plot on
    :param df: DataFrame containing sales, predictions, baseline, and filled target data for the entity
    :param entity: The entity to plot (e.g., product, store)
    :param group_col: The column name used for grouping (e.g., entity, store, product)
    :param cutoff: The cutoff date for the analysis
    :param baseline_col: The column name for baseline data
    :param target_col: The column name for target data (e.g., sales)
    :param title: Title for the plot
    """
    # Plot sales, predictions, baseline, and filled target for the selected entity
    ax.plot(df['date'], df[target_col], label=f'{entity} (Sales)', color='tab:blue')
    ax.plot(df['date'], df['prediction'], label=f'{entity} (Prediction)', color='tab:green', linestyle='--')
    ax.plot(df['date'], df[baseline_col], label=f'{entity} (Baseline)', color='tab:red', linestyle='--', alpha=0.7)

    # Add a vertical line for the cutoff date
    ax.axvline(x=pd.to_datetime(cutoff), color='black', linestyle='-', label='Cutoff Date')

    ax.set_xlabel('Date', fontsize=10)
    ax.tick_params(axis='y', labelsize=8)

    # Format x-axis for dates, showing every 2 months
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)

    # Set the title and legend
    ax.set_title(f'{group_col.upper()} {entity} - Cutoff {cutoff}', fontsize=12, fontweight='bold')
    ax.legend(fontsize=6, frameon=True, loc='upper left')

# Function to process and plot data for each cutoff and entity
def process_and_plot(df, group_col, baseline_col='baseline', target_col='sales', top_n=1, title=''):
    """
    Process and plot sales, predictions, and baseline data for each cutoff and entity in a 2-column layout.

    :param df: The input DataFrame containing sales, predictions, and baseline data
    :param group_col: The column name to group the data by (e.g., entity, store, product)
    :param baseline_col: The column name for the baseline data (default is 'baseline')
    :param target_col: The column name for target data (default is 'sales')
    :param top_n: The number of top entities to visualize based on total sales
    :param title: Title for the plot
    """
    # Ensure date is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Unique cutoffs
    cutoffs = df['cutoff'].unique()

    # Select top N entities based on total sales
    top_entities = df.groupby(group_col)[target_col].sum().nlargest(top_n).index

    # Number of subplots: rows for the plot grid
    total_plots = len(top_entities) * len(cutoffs)
    rows = ceil(total_plots / 2)

    # Create the figure and axes with the required number of subplots
    fig, axes = plt.subplots(rows, 2, figsize=(16, rows * 4))
    axes = axes.flatten()

    # Loop over each top entity
    plot_idx = 0
    for entity in top_entities:
        for cutoff in cutoffs:
            # Filter data for the current cutoff and the specific entity
            df_cutoff = df[(df['cutoff'] == cutoff) & (df[group_col] == entity)]

            # Group data by date, summing sales, predictions, and baselines
            df_grouped = df_cutoff.groupby(['date']).agg({
                target_col: 'sum',
                'prediction': 'sum',
                baseline_col: 'sum'
            }).reset_index()

            # Create a plot on the respective axis
            create_single_axis_plot(axes[plot_idx], df_grouped, entity, group_col, cutoff, baseline_col, target_col, title)
            plot_idx += 1

            # Ensure the figure fits well within the layout
            plt.tight_layout()

            # Optional sleep between plots for colab
            sleep(0.5)

    # Show the entire figure with all subplots
    plt.show()

# Prepare submission
def prepare_submission(df, value_to_pivot):
    """
    Prepares a submission DataFrame by filtering and pivoting the given DataFrame.

    This function filters the DataFrame for the latest cutoff date and the test samples,
    then pivots it so that unique dates become columns with corresponding values
    from the specified column (baseline, prediction, or others).
    """

    # Ensure the 'date' and 'cutoff' columns are in datetime format
    df['date'] = pd.to_datetime(df['date'])
    df['cutoff'] = pd.to_datetime(df['cutoff'])

    # Filter for the latest cutoff
    latest_cutoff = df['cutoff'].max()
    print(f"Latest cutoff date selected: {latest_cutoff}")  # Print the latest cutoff date

    # Filter for rows with the latest cutoff and 'test' samples
    filtered_df = df[(df['cutoff'] == latest_cutoff) & (df['sample'] == 'test')]

    # Filter for rows where 'date' is after the latest cutoff
    filtered_df = filtered_df[filtered_df['date'] > latest_cutoff]

    # Select relevant columns
    filtered_df = filtered_df[['client', 'warehouse', 'product', 'date', value_to_pivot]]

    # Pivot the DataFrame wider
    pivoted_df = filtered_df.pivot_table(index=['client', 'warehouse', 'product'],
                                          columns='date',
                                          values=value_to_pivot,
                                          aggfunc='first').reset_index()

    # Flatten the columns
    pivoted_df.columns.name = None  # Remove the index name
    pivoted_df.columns = [str(col.date()) if isinstance(col, pd.Timestamp) else col for col in pivoted_df.columns]

    return pivoted_df