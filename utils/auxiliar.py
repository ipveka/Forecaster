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
    ax, df, entity, group_col, cutoff, baseline_col, target_col, title
):
    """
    Create an enhanced single-axis plot for the specified entity.

    :param ax: The axis to plot on
    :param df: DataFrame containing sales, predictions, baseline, and filled target data
    :param entity: The entity to plot (e.g., product, store)
    :param group_col: The column name used for grouping
    :param cutoff: The cutoff date for the analysis
    :param baseline_col: The column name for baseline data
    :param target_col: The column name for target data
    :param title: Title for the plot
    """
    cutoff_date = pd.to_datetime(cutoff)
    
    # Separate pre and post cutoff data
    pre_cutoff = df[df["date"] <= cutoff_date]
    post_cutoff = df[df["date"] > cutoff_date]
    
    # Plot actual sales
    ax.plot(df["date"], df[target_col], label="Actual Sales", 
            color="#2E86AB", linewidth=2.5, zorder=4)
    
    # Plot model_prediction (if exists) - original model output before guardrail
    if "model_prediction" in df.columns and len(post_cutoff) > 0:
        # Connect last pre-cutoff point to first post-cutoff point
        connection_df = pd.concat([pre_cutoff.tail(1), post_cutoff])
        ax.plot(connection_df["date"], connection_df["model_prediction"], 
                color="#F77F00", linewidth=1.5, linestyle="-.", alpha=0.6, zorder=2)
        ax.plot(post_cutoff["date"], post_cutoff["model_prediction"], 
                label="Model Prediction", color="#F77F00", linewidth=1.5, 
                linestyle="-.", alpha=0.6, zorder=2)
    
    # Plot prediction (final prediction, potentially with guardrail applied)
    if len(post_cutoff) > 0:
        # Connect last pre-cutoff point to first post-cutoff point
        connection_df = pd.concat([pre_cutoff.tail(1), post_cutoff])
        ax.plot(connection_df["date"], connection_df["prediction"], 
                color="#06A77D", linewidth=2, linestyle="--", alpha=0.9, zorder=3)
        ax.plot(post_cutoff["date"], post_cutoff["prediction"], 
                label="Final Prediction", color="#06A77D", linewidth=2, 
                linestyle="--", alpha=0.9, zorder=3)
    
    # Plot baseline
    ax.plot(df["date"], df[baseline_col], label="Baseline", 
            color="#D62828", linewidth=1.8, linestyle=":", 
            alpha=0.7, zorder=1)
    
    # Add shaded region for forecast period
    if len(post_cutoff) > 0:
        ax.axvspan(cutoff_date, df["date"].max(), 
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
    
    # Title
    ax.set_title(f"{group_col.upper()}: {entity} | Cutoff: {cutoff_date.strftime('%Y-%m-%d')}", 
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
    df, group_col, baseline_col="baseline", target_col="sales", 
    top_n=1, title="", figsize_per_row=4.5
):
    """
    Process and plot sales, predictions, and baseline data with enhanced visualization.

    :param df: Input DataFrame with sales, predictions, and baseline data
    :param group_col: Column name to group by (e.g., entity, store, product)
    :param baseline_col: Column name for baseline data (default: 'baseline')
    :param target_col: Column name for target data (default: 'sales')
    :param top_n: Number of top entities to visualize based on total sales
    :param title: Overall title for the plot
    :param figsize_per_row: Height per row of subplots (default: 4.5)
    """
    # Ensure date is in datetime format
    df["date"] = pd.to_datetime(df["date"])
    
    # Get unique cutoffs
    cutoffs = sorted(df["cutoff"].unique())
    
    # Select top N entities based on total sales
    top_entities = df.groupby(group_col)[target_col].sum().nlargest(top_n).index.tolist()
    
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
            # Filter data
            df_cutoff = df[(df["cutoff"] == cutoff) & (df[group_col] == entity)]
            
            # Group by date
            agg_dict = {target_col: "sum", "prediction": "sum", baseline_col: "sum"}
            
            # Add model_prediction to aggregation if it exists
            if "model_prediction" in df_cutoff.columns:
                agg_dict["model_prediction"] = "sum"
            
            df_grouped = (
                df_cutoff.groupby("date")
                .agg(agg_dict)
                .reset_index()
            )
            
            # Create plot
            if plot_idx < len(axes):
                create_single_axis_plot(
                    axes[plot_idx], df_grouped, entity, group_col,
                    cutoff, baseline_col, target_col, title
                )
            plot_idx += 1
    
    # Hide unused subplots
    for idx in range(plot_idx, len(axes)):
        axes[idx].set_visible(False)
    
    # Adjust layout
    plt.tight_layout()
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
    df["date"] = pd.to_datetime(df["date"])
    df["cutoff"] = pd.to_datetime(df["cutoff"])

    # Filter for the latest cutoff
    latest_cutoff = df["cutoff"].max()
    print(
        f"Latest cutoff date selected: {latest_cutoff}"
    )  # Print the latest cutoff date

    # Filter for rows with the latest cutoff and 'test' samples
    filtered_df = df[(df["cutoff"] == latest_cutoff) & (df["sample"] == "test")]

    # Filter for rows where 'date' is after the latest cutoff
    filtered_df = filtered_df[filtered_df["date"] > latest_cutoff]

    # Select relevant columns
    filtered_df = filtered_df[
        ["client", "warehouse", "product", "date", value_to_pivot]
    ]

    # Pivot the DataFrame wider
    pivoted_df = filtered_df.pivot_table(
        index=["client", "warehouse", "product"],
        columns="date",
        values=value_to_pivot,
        aggfunc="first",
    ).reset_index()

    # Flatten the columns
    pivoted_df.columns.name = None  # Remove the index name
    pivoted_df.columns = [
        str(col.date()) if isinstance(col, pd.Timestamp) else col
        for col in pivoted_df.columns
    ]

    return pivoted_df
    