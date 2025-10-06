# Standard library imports
import numpy as np
import pandas as pd
import warnings
from math import ceil

# Third-party imports
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Unpivot data
def unpivot_data(df, id_vars, var_name="Date", value_name="Price"):
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
    df_unpiv = pd.melt(df, id_vars=id_vars, var_name=var_name, value_name=value_name)

    # Convert the 'Date' column from string to datetime format
    return df_unpiv


# Function to create a single axis plot for a specific entity
def create_single_axis_plot(
    ax, df, entity, group_col, cutoff, date_col="date", baseline_col="baseline", 
    target_col="sales", title=""
):
    """
    Create an enhanced single-axis plot for the specified entity.

    :param ax: The axis to plot on
    :param df: DataFrame containing sales, predictions, baseline, and filled target data
    :param entity: The entity to plot (e.g., product, store)
    :param group_col: The column name used for grouping
    :param cutoff: The cutoff date for the analysis
    :param date_col: The column name for date data (default: 'date')
    :param baseline_col: The column name for baseline data (default: 'baseline')
    :param target_col: The column name for target data (default: 'sales')
    :param title: Title for the plot (default: '')
    """
    cutoff_date = pd.to_datetime(cutoff)
    
    # Separate pre and post cutoff data
    pre_cutoff = df[df[date_col] <= cutoff_date]
    post_cutoff = df[df[date_col] > cutoff_date]
    
    # Plot actual sales
    ax.plot(df[date_col], df[target_col], label="Actual Sales", 
            color="#2E86AB", linewidth=2.5, zorder=4)
    
    # Plot model_prediction (if exists) - original model output before guardrail
    if "model_prediction" in df.columns and len(post_cutoff) > 0:
        # Connect last pre-cutoff point to first post-cutoff point
        connection_df = pd.concat([pre_cutoff.tail(1), post_cutoff])
        ax.plot(connection_df[date_col], connection_df["model_prediction"], 
                color="#F77F00", linewidth=1.5, linestyle="-.", alpha=0.6, zorder=2)
        ax.plot(post_cutoff[date_col], post_cutoff["model_prediction"], 
                label="Model Prediction", color="#F77F00", linewidth=1.5, 
                linestyle="-.", alpha=0.6, zorder=2)
    
    # Plot prediction (final prediction, potentially with guardrail applied)
    if len(post_cutoff) > 0:
        # Connect last pre-cutoff point to first post-cutoff point
        connection_df = pd.concat([pre_cutoff.tail(1), post_cutoff])
        ax.plot(connection_df[date_col], connection_df["prediction"], 
                color="#06A77D", linewidth=2, linestyle="--", alpha=0.9, zorder=3)
        ax.plot(post_cutoff[date_col], post_cutoff["prediction"], 
                label="Final Prediction", color="#06A77D", linewidth=2, 
                linestyle="--", alpha=0.9, zorder=3)
    
    # Plot prediction_ensemble (if exists) - smart ensemble prediction
    if "prediction_ensemble" in df.columns and len(post_cutoff) > 0:
        # Connect last pre-cutoff point to first post-cutoff point
        connection_df = pd.concat([pre_cutoff.tail(1), post_cutoff])
        ax.plot(connection_df[date_col], connection_df["prediction_ensemble"], 
                color="#8B5CF6", linewidth=2.5, linestyle="-", alpha=0.8, zorder=4)
        ax.plot(post_cutoff[date_col], post_cutoff["prediction_ensemble"], 
                label="Smart Ensemble", color="#8B5CF6", linewidth=2.5, 
                linestyle="-", alpha=0.8, zorder=4)
    
    # Plot baseline
    ax.plot(df[date_col], df[baseline_col], label="Baseline", 
            color="#D62828", linewidth=1.8, linestyle=":", 
            alpha=0.7, zorder=1)
    
    # Add shaded region for forecast period
    if len(post_cutoff) > 0:
        ax.axvspan(cutoff_date, df[date_col].max(), 
                   alpha=0.1, color='gray', label='Forecast Period')
    
    # Add vertical line for cutoff
    ax.axvline(x=cutoff_date, color="#333333", linewidth=2, 
               linestyle="-", label="Cutoff Date", alpha=0.8, zorder=4)
    
    # Styling
    ax.set_xlabel("Date", fontsize=11, fontweight='bold')
    ax.set_ylabel(target_col.capitalize(), fontsize=11, fontweight='bold')
    ax.tick_params(axis='both', labelsize=9)
    
    # Format x-axis
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.tick_params(axis='x', rotation=45)
    
    # Title - handle group_col as string or list
    group_label = group_col.upper() if isinstance(group_col, str) else "-".join([col.upper() for col in group_col])
    entity_label = str(entity) if not isinstance(entity, tuple) else "-".join([str(e) for e in entity])
    ax.set_title(f"{group_label}: {entity_label} | Cutoff: {cutoff_date.strftime('%Y-%m-%d')}", 
                fontsize=12, fontweight='bold', pad=12)
    
    # Legend with better positioning
    ax.legend(fontsize=8, frameon=True, loc='best', 
             framealpha=0.95, edgecolor='gray')
    
    # Add grid
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Improve spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')

# Process and plot
def process_and_plot(
    df, group_col, date_col="date", baseline_col="baseline", target_col="sales", 
    top_n=1, title="", figsize_per_row=4.5
):
    """
    Process and plot sales, predictions, and baseline data with enhanced visualization.

    :param df: Input DataFrame with sales, predictions, and baseline data
    :param group_col: Column name(s) to group by. Can be a string (e.g., 'product') or list (e.g., ['store', 'product'])
    :param date_col: Column name for date data (default: 'date')
    :param baseline_col: Column name for baseline data (default: 'baseline')
    :param target_col: Column name for target data (default: 'sales')
    :param top_n: Number of top entities to visualize based on total sales
    :param title: Overall title for the plot
    :param figsize_per_row: Height per row of subplots (default: 4.5)
    """
    # Ensure group_col is a list for consistent handling
    group_cols = [group_col] if isinstance(group_col, str) else group_col
    
    # Ensure date is in datetime format
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get unique cutoffs
    cutoffs = sorted(df["cutoff"].unique())
    
    # Select top N entities based on total sales
    top_entities = df.groupby(group_cols)[target_col].sum().nlargest(top_n).index.tolist()
    
    # Convert single values to tuples for consistent handling
    if isinstance(group_col, str):
        top_entities = [(entity,) if not isinstance(entity, tuple) else entity for entity in top_entities]
    
    # Calculate layout
    total_plots = len(top_entities) * len(cutoffs)
    rows = ceil(total_plots / 2)
    
    # Create figure
    fig, axes = plt.subplots(rows, 2, figsize=(18, rows * figsize_per_row))
    
    # Handle single subplot case
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16, fontweight='bold', y=1.002)
    
    # Loop over entities and cutoffs
    plot_idx = 0
    for entity in top_entities:
        for cutoff in cutoffs:
            # Filter data based on group columns
            if isinstance(group_col, str):
                # Single column - entity is a tuple with one element
                mask = (df["cutoff"] == cutoff) & (df[group_col] == entity[0])
            else:
                # Multiple columns - entity is a tuple
                mask = (df["cutoff"] == cutoff)
                for col, val in zip(group_cols, entity):
                    mask &= (df[col] == val)
            
            df_cutoff = df[mask]
            
            # Group by date
            agg_dict = {target_col: "sum", "prediction": "sum", baseline_col: "sum"}
            
            # Add model_prediction to aggregation if it exists
            if "model_prediction" in df_cutoff.columns:
                agg_dict["model_prediction"] = "sum"
            
            # Add prediction_ensemble to aggregation if it exists
            if "prediction_ensemble" in df_cutoff.columns:
                agg_dict["prediction_ensemble"] = "sum"
            
            df_grouped = (
                df_cutoff.groupby(date_col)
                .agg(agg_dict)
                .reset_index()
            )
            
            # Create plot
            if plot_idx < len(axes):
                create_single_axis_plot(
                    axes[plot_idx], df_grouped, entity, group_col,
                    cutoff, date_col, baseline_col, target_col, title
                )
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
    plt.show()
    