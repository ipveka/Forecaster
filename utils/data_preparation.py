# Standard library imports
import gc
import os
import warnings
from itertools import product
from pathlib import Path

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# ML library imports
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Pd options
pd.options.mode.chained_assignment = None


# Data preparation class
class DataPreparation:
    # Init
    def __init__(self):
        pass

    def run_data_preparation(
        self,
        df,
        group_cols,
        date_col,
        target,
        horizon=None,
        freq=None,
        complete_dataframe=True,
        smoothing=True,
        dp_window_size=None,
        n_cutoffs=1,
    ):
        """
        Main function to prepare the data by calling all internal functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        group_cols (list): Columns to group the data
        date_col (str): Column containing dates
        target (str): Target variable
        horizon (int, optional): Forecasting horizon. If None, it will be determined by frequency.
        freq (str, optional): Frequency of the data. If None, it will be auto-detected.
        complete_dataframe (bool): Whether to use complete_dataframe function
        smoothing (bool): Whether to smooth signals or not
        dp_window_size (int, optional): Window size for smoothing. If None, it will be determined by frequency.
        n_cutoffs (int): Number of cutoffs for backtesting

        Returns:
        pd.DataFrame: Prepared DataFrame
        """
        # Print header
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)
        
        # Log initial dataset info
        initial_rows = len(df)
        initial_cols = len(df.columns)
        n_groups = df[group_cols].drop_duplicates().shape[0] if isinstance(group_cols, list) else df[group_cols].nunique()
        print(f"\n📊 Input Dataset:")
        print(f"   • Rows: {initial_rows:,}")
        print(f"   • Columns: {initial_cols}")
        print(f"   • Groups: {n_groups:,}")
        print(f"   • Date column: '{date_col}'")
        print(f"   • Target: '{target}'")

        # Convert date column to datetime once at the beginning
        print(f"\n🔄 Converting '{date_col}' to datetime format...")
        df[date_col] = pd.to_datetime(df[date_col])
        date_range = f"{df[date_col].min().date()} to {df[date_col].max().date()}"
        print(f"   ✓ Date range: {date_range}")
        
        # Show date range for non-zero, non-NaN target values
        valid_target_mask = (df[target] > 0) & (df[target].notna())
        if valid_target_mask.any():
            valid_target_df = df[valid_target_mask]
            target_date_range = f"{valid_target_df[date_col].min().date()} to {valid_target_df[date_col].max().date()}"
            n_valid_records = valid_target_mask.sum()
            print(f"   ✓ Target > 0 & not NaN: {n_valid_records:,} records ({target_date_range})")
        else:
            print(f"   ⚠️  Warning: No records found where target > 0 and not NaN")

        # Auto-detect frequency if not provided
        if freq is None:
            print(f"\n🔍 Auto-detecting frequency...")
            freq = self.detect_frequency(df, date_col)
            print(f"   ✓ Detected frequency: {freq}")
        else:
            print(f"\n📅 Using specified frequency: {freq}")

        # Get frequency-specific parameters
        from utils.forecaster_utils import get_frequency_params

        freq_params = get_frequency_params(freq)

        # Use frequency-specific parameters if not provided by user
        if horizon is None:
            horizon = freq_params["horizon"]
            print(f"   ✓ Using frequency-based horizon: {horizon} periods")
        else:
            print(f"   ✓ Using specified horizon: {horizon} periods")

        if dp_window_size is None:
            dp_window_size = freq_params["dp_window_size"]
            print(f"   ✓ Using frequency-based smoothing window: {dp_window_size} periods")
        else:
            print(f"   ✓ Using specified smoothing window: {dp_window_size} periods")

        # Complete logic
        if complete_dataframe:
            print(f"\n🔧 Filling missing dates for each group...")
            rows_before = len(df)
            df = self.complete(df, group_cols, date_col, freq)
            rows_after = len(df)
            rows_added = rows_after - rows_before
            print(f"   ✓ Added {rows_added:,} rows to complete date ranges")
            print(f"   ✓ Total rows: {rows_after:,}")
        else:
            print(f"\n⏭️  Skipping date completion (complete_dataframe=False)")

        # Find all numeric columns to be treated as signals
        signal_cols = [col for col in df.select_dtypes(include=["float64"]).columns]
        print(f"\n📈 Identified {len(signal_cols)} signal column(s):")
        for col in signal_cols:
            non_null_pct = (df[col].notna().sum() / len(df)) * 100
            print(f"   • {col} ({non_null_pct:.1f}% non-null)")

        # Smoothing of signals
        if smoothing and signal_cols:
            print(f"\n🔄 Applying smoothing to {len(signal_cols)} signal(s)...")
            df = self.smoothing(df, group_cols, date_col, signal_cols, dp_window_size)
            print(f"   ✓ Applied moving average with window size: {dp_window_size}")
            print(f"   ✓ Created 'filled_*' columns for each signal")
        elif not smoothing:
            print(f"\n⏭️  Skipping smoothing (smoothing=False)")
        else:
            print(f"\n⚠️  No signal columns found for smoothing")

        # Get last cutoffs
        print(f"\n📅 Creating {n_cutoffs} cutoff(s) for backtesting...")
        cutoff_dates = self.get_first_dates_last_n_months(df, date_col, target, n_cutoffs)
        print(f"   ✓ Cutoff dates (based on dates with valid target):")
        for i, cutoff_date in enumerate(cutoff_dates, 1):
            cutoff_label = " [LATEST - Used for future forecasting]" if i == 1 else ""
            print(f"      {i}. {pd.to_datetime(cutoff_date).date()}{cutoff_label}")

        # Create backtesting
        print(f"\n🔀 Creating train/test splits...")
        rows_before = len(df)
        df = self.create_backtesting_df(df, date_col, cutoff_dates)
        rows_after = len(df)
        print(f"   ✓ Expanded dataset from {rows_before:,} to {rows_after:,} rows")
        train_rows = (df["sample"] == "train").sum()
        test_rows = (df["sample"] == "test").sum()
        print(f"   ✓ Train samples: {train_rows:,} ({train_rows/rows_after*100:.1f}%)")
        print(f"   ✓ Test samples: {test_rows:,} ({test_rows/rows_after*100:.1f}%)")

        # Add horizon for last cutoff
        print(f"\n🎯 Adding forecast horizon for latest cutoff...")
        rows_before = len(df)
        df = self.add_horizon_last_cutoff(df, group_cols, date_col, horizon, freq)
        rows_after = len(df)
        rows_added = rows_after - rows_before
        if rows_added > 0:
            print(f"   ✓ Added {rows_added:,} rows for {horizon}-period forecast horizon")
        else:
            print(f"   ✓ Horizon already complete (no rows added)")

        # Final formatting
        print(f"\n🔧 Final formatting...")
        df["cutoff"] = pd.to_datetime(df["cutoff"])
        print(f"   ✓ Converted cutoff column to datetime")

        # Final summary
        print(f"\n" + "=" * 70)
        print(f"✅ DATA PREPARATION COMPLETED")
        print(f"=" * 70)
        print(f"   • Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   • Cutoffs: {n_cutoffs}")
        print(f"   • Horizon: {horizon} periods")
        print(f"   • Frequency: {freq}")
        print("=" * 70 + "\n")
        
        return df

    # Detect frequency
    def detect_frequency(self, df, date_col):
        """
        Automatically detect the frequency of time series data based on the date column.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.

        Returns:
        str: The detected frequency as a pandas frequency string (e.g., 'D', 'W', 'M', 'Q', 'Y').
        """
        # Make sure the date column is datetime
        dates = pd.Series(df[date_col].unique()).sort_values()

        # Calculate time differences between consecutive dates
        time_diffs = dates.diff().dropna()

        if len(time_diffs) == 0:
            # Default to daily if we can't determine
            return "D"

        # Calculate the most common time difference in days
        median_diff_days = time_diffs.median().days

        # Get the most common difference in days, defaulting to median if empty
        day_counts = time_diffs.dt.days.value_counts()
        if day_counts.empty:
            most_common_diff = median_diff_days
        else:
            most_common_diff = day_counts.index[0]

        # Map the days difference to standard pandas frequency strings
        if most_common_diff <= 1:
            # Daily
            return "D"
        elif 2 <= most_common_diff <= 3:
            # Business days
            return "B"
        elif 6 <= most_common_diff <= 8:
            # Weekly - check if it's consistently on the same day of week
            dow_counts = dates.dt.dayofweek.value_counts()
            most_common_dow = dow_counts.idxmax()
            dow_pct = dow_counts.max() / dow_counts.sum()

            if dow_pct > 0.7:  # If more than 70% of dates fall on the same day of week
                dow_map = {
                    0: "W-MON",
                    1: "W-TUE",
                    2: "W-WED",
                    3: "W-THU",
                    4: "W-FRI",
                    5: "W-SAT",
                    6: "W-SUN",
                }
                return dow_map[most_common_dow]
            else:
                return "W"  # Generic weekly
        elif 28 <= most_common_diff <= 31:
            # Monthly - check if it's consistently on the same day of month
            dom_counts = dates.dt.day.value_counts()
            if dom_counts.empty:
                return "M"  # Default to month-end if empty

            dom_pct = dom_counts.max() / dom_counts.sum()

            if dom_pct > 0.7:  # If more than 70% of dates fall on the same day of month
                return "MS" if dates.dt.day.mode()[0] <= 5 else "M"
            else:
                return "M"  # End of month
        elif 89 <= most_common_diff <= 93:
            return "Q"  # Quarterly
        elif 180 <= most_common_diff <= 186:
            return "SM"  # Semi-month
        elif 364 <= most_common_diff <= 366:
            return "Y"  # Yearly
        else:
            # For unusual frequencies, return a custom frequency based on days
            return f"{int(most_common_diff)}D"

    # Complete dataframe (optimized for performance)
    def complete(self, df, group_cols, date_col, freq="W-MON"):
        """
        Ensure that each group has all dates with a specified frequency from its first to its last date.
        Missing rows will be filled with NaN values. Optimized for performance with vectorized operations.

        Parameters:
        df (pd.DataFrame): The original DataFrame.
        group_cols (list): List of columns to group by (e.g. ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        freq (str): Frequency string for the date range (e.g., 'W-MON' for weekly on Mondays).

        Returns:
        pd.DataFrame: The completed DataFrame with all required date ranges for each group.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Group by the specified columns and compute min and max dates for each group
        group_ranges = (
            df.groupby(group_cols)[date_col].agg(["min", "max"]).reset_index()
        )

        # Prepare to create a list of DataFrames for each group's date range
        all_dfs = []

        # Create date ranges for each group more efficiently
        for _, row in group_ranges.iterrows():
            # Extract group values and date range
            group_values = {col: row[col] for col in group_cols}
            date_range = pd.date_range(start=row["min"], end=row["max"], freq=freq)

            # Create DataFrame with all dates for this group
            group_df = pd.DataFrame({date_col: date_range})

            # Add group columns
            for col, val in group_values.items():
                group_df[col] = val

            # Add to our list of DataFrames
            all_dfs.append(group_df)

        # Combine all groups' date ranges into one DataFrame
        if all_dfs:
            completed_df = pd.concat(all_dfs, ignore_index=True)

            # Merge with original data efficiently
            completed_df = pd.merge(
                completed_df, df, on=group_cols + [date_col], how="left"
            )
        else:
            # If no date ranges, return original DataFrame
            completed_df = df.copy()

        return completed_df

    # Smoothing (optimized for performance)
    def smoothing(
        self, df, group_columns, date_column, signal_columns, dp_window_size=13
    ):
        """
        Fill missing values in signal columns using a rolling moving average, calculated for each group.
        Optimized version using vectorized operations for better performance.

        Parameters:
        df (pd.DataFrame): Input dataframe containing the data.
        group_columns (list): List of columns to group the data by (e.g., ['client', 'warehouse']).
        date_column (str): Column name representing the date (used to sort the data).
        signal_columns (list): List of signal columns where missing values will be filled (e.g., ['sales', 'price']).
        dp_window_size (int): The window size for the moving average. Default is 13.

        Returns:
        pd.DataFrame: A new dataframe with additional columns for each signal, where missing values have been filled.
        """
        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Sort by the grouping columns and the date column to apply the rolling average in order
        df_copy = df_copy.sort_values(by=group_columns + [date_column])

        # Process each signal column with vectorized operations
        for signal_column in signal_columns:
            # Create a new column name for the filled signal
            filled_signal_column = f"filled_{signal_column}"

            # Copy the original signal values into the new column
            df_copy[filled_signal_column] = df_copy[signal_column]

            # Use transform to apply the rolling mean to each group in one operation
            # This is more efficient than looping through each group
            df_copy[filled_signal_column] = df_copy.groupby(group_columns)[
                signal_column
            ].transform(
                lambda x: x.fillna(
                    x.rolling(window=dp_window_size, min_periods=1).mean()
                )
            )

            # Ensure that the filled values are greater than or equal to zero
            df_copy[filled_signal_column] = df_copy[filled_signal_column].clip(lower=0)

        # Return the modified dataframe with the new columns where missing values are filled
        return df_copy

    # Latest dates
    def get_latest_n_dates(self, df, date_col, target, n_cutoffs):
        """
        Get the latest n distinct dates from the DataFrame,
        considering only dates where the target is > 0 and not NaN.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        target (str): The name of the target column.
        n_cutoffs (int): The number of distinct dates to retrieve.

        Returns:
        pd.Series: A series containing the latest n distinct dates.
        """

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to only dates where target is valid (> 0 and not NaN)
        valid_target_df = df[(df[target] > 0) & (df[target].notna())].copy()
        
        if valid_target_df.empty:
            raise ValueError(f"No valid target values found (target > 0 and not NaN) in column '{target}'")

        # Get distinct dates, sorted in descending order
        distinct_dates = valid_target_df[date_col].drop_duplicates().sort_values(ascending=False)

        # Select the latest n distinct dates
        latest_dates = distinct_dates.head(n_cutoffs)

        return latest_dates

    # Latest month first weeks
    def get_first_dates_last_n_months(self, df, date_col, target, n_cutoffs):
        """
        Get cutoff dates for backtesting, ensuring the latest valid date is always included.
        For n_cutoffs > 1, also includes the first date of previous months.
        All dates are filtered to only include dates where target is > 0 and not NaN.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        target (str): The name of the target column.
        n_cutoffs (int): The number of cutoffs to retrieve.

        Returns:
        list: A list containing cutoff dates, with the latest valid date always first.
        """

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Filter to only dates where target is valid (> 0 and not NaN)
        valid_target_df = df[(df[target] > 0) & (df[target].notna())].copy()
        
        if valid_target_df.empty:
            raise ValueError(f"No valid target values found (target > 0 and not NaN) in column '{target}'")

        # Always get the latest date with valid target (for forecasting)
        latest_date = valid_target_df[date_col].max()
        cutoff_dates = [latest_date]
        
        # If we need more cutoffs, get first dates from previous months
        if n_cutoffs > 1:
            # Group by year and month, then get the first date for each month
            first_dates_per_month = valid_target_df.groupby(valid_target_df[date_col].dt.to_period("M"))[
                date_col
            ].min()
            
            # Get the latest n-1 months (excluding the current month if latest_date is already included)
            # Sort in descending order and take enough to fill n_cutoffs
            sorted_first_dates = first_dates_per_month.sort_index(ascending=False)
            
            for first_date in sorted_first_dates:
                # Skip if this date is the same as latest_date (avoid duplicates)
                if first_date != latest_date and len(cutoff_dates) < n_cutoffs:
                    cutoff_dates.append(first_date)
                    
                if len(cutoff_dates) >= n_cutoffs:
                    break

        return cutoff_dates

    # Create cutoffs (optimized for performance)
    def create_backtesting_df(self, df, date_col, cutoff_dates):
        """
        Create train/test splits based on cutoff dates and concatenate results into a single DataFrame.
        Optimized version using vectorized operations and cross-join for better performance.

        Parameters:
        df (pd.DataFrame): The input DataFrame with data to be split.
        date_col (str): The name of the date column.
        cutoff_dates (pd.Series): Series containing the cutoff dates.

        Returns:
        pd.DataFrame: A DataFrame with additional columns 'cutoff' and 'sample' indicating train/test splits.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Convert cutoff_dates to DataFrame for cross join
        cutoffs_df = pd.DataFrame({"cutoff": cutoff_dates})

        # Add a key column for cross join
        df_with_key = df.assign(_key=1)
        cutoffs_with_key = cutoffs_df.assign(_key=1)

        # Perform cross join to create all combinations of rows and cutoffs
        combined = pd.merge(df_with_key, cutoffs_with_key, on="_key").drop(
            "_key", axis=1
        )

        # Set the 'sample' column based on date comparison with cutoff
        # Using numpy for vectorized operations instead of apply
        combined["sample"] = np.where(
            combined[date_col] > combined["cutoff"],
            "test",
            np.where(combined[date_col] == combined["cutoff"], "test", "train"),
        )

        return combined

    # Add full horizon (optimized for performance)
    def add_horizon_last_cutoff(self, df, group_cols, date_col, horizon, freq="W"):
        """
        Ensure that for each group in the latest cutoff, we have the specified horizon (e.g., 13 weeks)
        after the cutoff date. If they don't exist, add them.
        All other data remains unchanged. Optimized version with vectorized operations.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        group_cols (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        horizon (int): The number of periods that should be covered after the cutoff date.
        freq (str): The frequency of the dates. Default is 'W' for weekly.

        Returns:
        pd.DataFrame: The updated DataFrame with any missing dates filled for the specified horizon in the latest cutoff.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Identify the latest cutoff
        latest_cutoff = df["cutoff"].max()

        # Split the dataframe into latest cutoff and the rest
        latest_df = df[df["cutoff"] == latest_cutoff]
        rest_df = df[df["cutoff"] != latest_cutoff]

        # Only proceed if we have data for the latest cutoff
        if latest_df.empty:
            return df

        # Get unique groups in the latest cutoff
        unique_groups = latest_df[group_cols].drop_duplicates()

        # Create a list to store new horizon rows
        horizon_rows = []

        # Process each group individually (avoiding the deprecated groupby-apply)
        for _, group_values in unique_groups.iterrows():
            # Create filter for this group
            group_filter = True
            for col, val in zip(group_cols, [group_values[col] for col in group_cols]):
                group_filter = group_filter & (latest_df[col] == val)

            # Get this group's data
            group_data = latest_df[group_filter]

            if len(group_data) > 0:
                # Get the cutoff date for this group
                cutoff_date = pd.to_datetime(group_data["cutoff"].iloc[0])

                # Generate the required date range starting from the cutoff date
                required_dates = pd.date_range(
                    start=cutoff_date, periods=horizon + 1, freq=freq
                )[1:]

                # Add each date as a new row with the group's data
                for date in required_dates:
                    # Check if this date already exists for this group
                    date_filter = group_filter & (latest_df[date_col] == date)
                    # Fixed the datetime64 any() warning
                    if len(latest_df[date_filter]) == 0:  # If no matching row exists
                        # Create a new row with this date and group values
                        new_row = {
                            date_col: date,
                            "sample": "test",
                            "cutoff": latest_cutoff,
                        }

                        # Add group column values
                        for col in group_cols:
                            new_row[col] = group_values[col]

                        horizon_rows.append(new_row)

        # Create DataFrame from horizon rows if we have any
        if horizon_rows:
            horizon_df = pd.DataFrame(horizon_rows)

            # Combine with latest cutoff data
            processed_latest = pd.concat([latest_df, horizon_df], ignore_index=True)

            # Combine with the rest of the data
            result = pd.concat([rest_df, processed_latest], ignore_index=True)
        else:
            # If no horizon rows needed, just return the original data
            result = df.copy()

        # Fill forward string/object columns
        string_columns = result.select_dtypes(include="object").columns

        # Sort the data properly before filling
        result = result.sort_values(by=group_cols + ["cutoff", date_col])

        # Fill forward within each group
        for col in string_columns:
            result[col] = result.groupby(group_cols)[col].transform(lambda x: x.ffill()).infer_objects(copy=False)

        # Also fill forward any filled_* columns to ensure no NaNs in the forecast horizon
        filled_columns = [col for col in result.columns if col.startswith("filled_")]
        for col in filled_columns:
            if col in result.columns:
                result[col] = result.groupby(group_cols)[col].transform(
                    lambda x: x.ffill()
                ).infer_objects(copy=False)

        return result
