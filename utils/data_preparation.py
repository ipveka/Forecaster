# General libraries
import pandas as pd
import numpy as np

# Utilities
from pathlib import Path
import psutil
import gc
import os

# Other libraries
from itertools import product

# Plots
import matplotlib.pyplot as plt

# ML
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor

# Warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# Pd options
pd.options.mode.chained_assignment = None

# Data preparation class
class DataPreparation:
    # Init
    def __init__(self):
        pass

    def run_data_preparation(self, df, group_cols, date_col, target, horizon, freq=None, complete_dataframe=True,
                             smoothing=True, ma_window_size=13, n_cutoffs=13):
        """
        Main function to prepare the data by calling all internal functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        group_cols (list): Columns to group the data
        date_col (str): Column containing dates
        target (str): Target variable
        horizon (int): Forecasting horizon
        freq (str, optional): Frequency of the data. If None, it will be auto-detected.
        complete_dataframe (bool): Whether to use complete_dataframe function
        smoothing (bool): Whether to smooth signals or not
        ma_window_size (int): Window size for smoothing
        n_cutoffs (int): Number of cutoffs for backtesting

        Returns:
        pd.DataFrame: Prepared DataFrame
        """
        # Start function
        print("Starting data preparation...")

        # Convert date column to datetime once at the beginning
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Auto-detect frequency if not provided
        if freq is None:
            freq = self.detect_frequency(df, date_col)
            print(f"Auto-detected frequency: {freq}")

        # Complete logic
        if complete_dataframe:
            df = self.complete(df, group_cols, date_col, freq)
            print("Completed DataFrame by filling in missing values.")

        # Find all numeric columns to be treated as signals
        signal_cols = [col for col in df.select_dtypes(include=['float64']).columns]
        print(f"Identified signal columns: {signal_cols}")

        # Smoothing of signals
        if smoothing:
            df = self.smoothing(df, group_cols, date_col, signal_cols, ma_window_size)
            print(f"Applied smoothing with a moving average window size of {ma_window_size}.")

        # Get last cutoffs
        cutoff_dates = self.get_first_dates_last_n_months(df, date_col, n_cutoffs)
        print(f"Identified cutoff dates for backtesting: {cutoff_dates}")

        # Create backtesting
        df = self.create_backtesting_df(df, date_col, cutoff_dates)
        print("Created backtesting DataFrame.")

        # Add horizon for last cutoff
        df = self.add_horizon_last_cutoff(df, group_cols, date_col, horizon, freq)
        print("Added forecasting horizon for the last cutoff.")

        # Final formatting
        df['cutoff'] = pd.to_datetime(df['cutoff'])

        # Final message and return
        print("Data preparation completed.")
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
            return 'D'
        
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
            return 'D'
        elif 2 <= most_common_diff <= 3:
            # Business days
            return 'B'
        elif 6 <= most_common_diff <= 8:
            # Weekly - check if it's consistently on the same day of week
            dow_counts = dates.dt.dayofweek.value_counts()
            most_common_dow = dow_counts.idxmax()
            dow_pct = dow_counts.max() / dow_counts.sum()
            
            if dow_pct > 0.7:  # If more than 70% of dates fall on the same day of week
                dow_map = {0: 'W-MON', 1: 'W-TUE', 2: 'W-WED', 3: 'W-THU', 4: 'W-FRI', 5: 'W-SAT', 6: 'W-SUN'}
                return dow_map[most_common_dow]
            else:
                return 'W'  # Generic weekly
        elif 28 <= most_common_diff <= 31:
            # Monthly - check if it's consistently on the same day of month
            dom_counts = dates.dt.day.value_counts()
            if dom_counts.empty:
                return 'M'  # Default to month-end if empty
                
            dom_pct = dom_counts.max() / dom_counts.sum()
            
            if dom_pct > 0.7:  # If more than 70% of dates fall on the same day of month
                return 'MS' if dates.dt.day.mode()[0] <= 5 else 'M'
            else:
                return 'M'  # End of month
        elif 89 <= most_common_diff <= 93:
            return 'Q'  # Quarterly
        elif 180 <= most_common_diff <= 186:
            return 'SM'  # Semi-month
        elif 364 <= most_common_diff <= 366:
            return 'Y'  # Yearly
        else:
            # For unusual frequencies, return a custom frequency based on days
            return f'{int(most_common_diff)}D'
    
    # Complete dataframe (optimized for performance)
    def complete(self, df, group_cols, date_col, freq='W-MON'):
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
        group_ranges = df.groupby(group_cols)[date_col].agg(['min', 'max']).reset_index()
        
        # Prepare to create a list of DataFrames for each group's date range
        all_dfs = []
        
        # Create date ranges for each group more efficiently
        for _, row in group_ranges.iterrows():
            # Extract group values and date range
            group_values = {col: row[col] for col in group_cols}
            date_range = pd.date_range(start=row['min'], end=row['max'], freq=freq)
            
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
            completed_df = pd.merge(completed_df, df, on=group_cols + [date_col], how='left')
        else:
            # If no date ranges, return original DataFrame
            completed_df = df.copy()
        
        return completed_df

    # Smoothing (optimized for performance)
    def smoothing(self, df, group_columns, date_column, signal_columns, ma_window_size=13):
        """
        Fill missing values in signal columns using a rolling moving average, calculated for each group.
        Optimized version using vectorized operations for better performance.

        Parameters:
        df (pd.DataFrame): Input dataframe containing the data.
        group_columns (list): List of columns to group the data by (e.g., ['client', 'warehouse']).
        date_column (str): Column name representing the date (used to sort the data).
        signal_columns (list): List of signal columns where missing values will be filled (e.g., ['sales', 'price']).
        ma_window_size (int): The window size for the moving average. Default is 13.

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
            filled_signal_column = f'filled_{signal_column}'

            # Copy the original signal values into the new column
            df_copy[filled_signal_column] = df_copy[signal_column]

            # Use transform to apply the rolling mean to each group in one operation
            # This is more efficient than looping through each group
            df_copy[filled_signal_column] = df_copy.groupby(group_columns)[signal_column].transform(
                lambda x: x.fillna(x.rolling(window=ma_window_size, min_periods=1).mean())
            )

            # Ensure that the filled values are greater than or equal to zero
            df_copy[filled_signal_column] = df_copy[filled_signal_column].clip(lower=0)

        # Return the modified dataframe with the new columns where missing values are filled
        return df_copy

    # Latest dates
    def get_latest_n_dates(self, df, date_col, n_cutoffs):
        """
        Get the latest n distinct dates from the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        n_cutoffs (int): The number of distinct dates to retrieve.

        Returns:
        pd.Series: A series containing the latest n distinct dates.
        """

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Get distinct dates, sorted in descending order
        distinct_dates = df[date_col].drop_duplicates().sort_values(ascending=False)

        # Select the latest n distinct dates
        latest_dates = distinct_dates.head(n_cutoffs)

        return latest_dates

    # Latest month first weeks
    def get_first_dates_last_n_months(self, df, date_col, n_cutoffs):
        """
        Get the first date for the last n months from the DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame.
        date_col (str): The name of the date column.
        n_cutoffs (int): The number of months to retrieve.

        Returns:
        list: A list containing the first date of each of the last n months.
        """

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Group by year and month, then get the first date for each month
        first_dates_per_month = df.groupby(df[date_col].dt.to_period('M'))[date_col].min()

        # Get the latest n months
        latest_first_dates = first_dates_per_month.sort_index(ascending=False).head(n_cutoffs)

        # Convert to list and return
        return latest_first_dates.tolist()

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
        cutoffs_df = pd.DataFrame({'cutoff': cutoff_dates})
        
        # Add a key column for cross join
        df_with_key = df.assign(_key=1)
        cutoffs_with_key = cutoffs_df.assign(_key=1)
        
        # Perform cross join to create all combinations of rows and cutoffs
        combined = pd.merge(df_with_key, cutoffs_with_key, on='_key').drop('_key', axis=1)
        
        # Set the 'sample' column based on date comparison with cutoff
        # Using numpy for vectorized operations instead of apply
        combined['sample'] = np.where(
            combined[date_col] > combined['cutoff'], 
            'test',
            np.where(
                combined[date_col] == combined['cutoff'],
                'test', 
                'train'
            )
        )
        
        return combined

    # Add full horizon (optimized for performance)
    def add_horizon_last_cutoff(self, df, group_cols, date_col, horizon, freq='W'):
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
        latest_cutoff = df['cutoff'].max()

        # Split the dataframe into latest cutoff and the rest
        latest_df = df[df['cutoff'] == latest_cutoff]
        rest_df = df[df['cutoff'] != latest_cutoff]

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
                cutoff_date = pd.to_datetime(group_data['cutoff'].iloc[0])
                
                # Generate the required date range starting from the cutoff date
                required_dates = pd.date_range(start=cutoff_date, periods=horizon + 1, freq=freq)[1:]
                
                # Add each date as a new row with the group's data
                for date in required_dates:
                    # Check if this date already exists for this group
                    date_filter = group_filter & (latest_df[date_col] == date)
                    # Fixed the datetime64 any() warning
                    if len(latest_df[date_filter]) == 0:  # If no matching row exists
                        # Create a new row with this date and group values
                        new_row = {date_col: date, 'sample': 'test', 'cutoff': latest_cutoff}
                        
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
        string_columns = result.select_dtypes(include='object').columns
        
        # Sort the data properly before filling
        result = result.sort_values(by=group_cols + ['cutoff', date_col])
        
        # Fill forward within each group
        for col in string_columns:
            result[col] = result.groupby(group_cols)[col].transform(lambda x: x.ffill())
        
        # Also fill forward any filled_* columns to ensure no NaNs in the forecast horizon
        filled_columns = [col for col in result.columns if col.startswith('filled_')]
        for col in filled_columns:
            if col in result.columns:
                result[col] = result.groupby(group_cols)[col].transform(lambda x: x.ffill())
        
        return result