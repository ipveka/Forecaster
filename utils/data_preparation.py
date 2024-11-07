# General libraries
import pandas as pd
import numpy as np
import warnings
import psutil
import gc
import os
from itertools import product
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lightgbm import LGBMRegressor
from sklearn.preprocessing import LabelEncoder

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None

# Data preparation class
class DataPreparation:
    # Init
    def __init__(self):
        pass

    def run_data_preparation(self, df, group_cols, date_col, target, horizon, freq, complete_dataframe=True,
                             smoothing=True, ma_window_size=13, n_cutoffs=13):
        """
        Main function to prepare the data by calling all internal functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        group_cols (list): Columns to group the data
        date_col (str): Column containing dates
        target (str): Target variable
        horizon (int): Forecasting horizon
        freq (str): Frequency of the data
        complete_dataframe (bool): Whether to use complete_dataframe function
        smoothing (bool): Whether to smooth signals or not
        ma_window_size (int): Window size for smoothing
        n_cutoffs (int): Number of cutoffs for backtesting

        Returns:
        pd.DataFrame: Prepared DataFrame
        """
        # Start function
        print("Starting data preparation...")

        # Complete logic
        if complete_dataframe:
            df = self.complete(df, group_cols, date_col), freq
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

    # Complete dataframe
    def complete(self, df, group_cols, date_col, freq='W-MON'):
        """
        Ensure that each group has all dates with a specified frequency from its first to its last date.
        Missing rows will be filled with NaN values.

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
        group_ranges.columns = group_cols + [f'{date_col}_min', f'{date_col}_max']

        # Initialize an empty DataFrame to store the result
        completed_df = pd.DataFrame()

        # Generate all required dates for each group based on the specified frequency
        for _, row in group_ranges.iterrows():
            start_date = row[f'{date_col}_min']
            end_date = row[f'{date_col}_max']
            dates = pd.date_range(start=start_date, end=end_date, freq=freq)

            # Create a DataFrame for this group with all dates
            temp_df = pd.DataFrame({
                **row.drop([f'{date_col}_min', f'{date_col}_max']).to_dict(),
                date_col: dates
            })

            # Append this to the completed DataFrame
            completed_df = pd.concat([completed_df, temp_df], ignore_index=True)

        # Merge with the original DataFrame
        completed_df = pd.merge(completed_df, df, on=group_cols + [date_col], how='left')

        return completed_df

    # Smoothing
    def smoothing(self, df, group_columns, date_column, signal_columns, ma_window_size=13):
        """
        Fill missing values in signal columns using a rolling moving average, calculated for each group.

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

        # Loop through each signal column
        for signal_column in signal_columns:
            # Create a new column name for the filled signal
            filled_signal_column = f'filled_{signal_column}'

            # Copy the original signal values into the new column
            df_copy[filled_signal_column] = df_copy[signal_column]

            # Group the dataframe by the specified group columns (e.g., by client and warehouse)
            for name, group in df_copy.groupby(group_columns):
                # Calculate the rolling mean for this group
                rolling_mean = group[signal_column].rolling(window=ma_window_size, min_periods=1).mean()

                # Fill missing values (NaN) in the signal column with the rolling mean
                df_copy.loc[group.index, filled_signal_column] = group[signal_column].fillna(rolling_mean)

                # Ensure that the filled values are greater than zero
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

    # Create cutoffs
    def create_backtesting_df(self, df, date_col, cutoff_dates):
        """
        Create train/test splits based on cutoff dates and concatenate results into a single DataFrame.

        Parameters:
        df (pd.DataFrame): The input DataFrame with 'client', 'warehouse', 'product', 'date', 'sales', 'price', 'filled_sales', 'filled_price'.
        date_col (str): The name of the date column.
        cutoff_dates (pd.Series): Series containing the cutoff dates.

        Returns:
        pd.DataFrame: A DataFrame with additional columns 'cutoff_date' and 'sample' indicating train/test splits.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Initialize a list to store the resulting DataFrames
        results = []

        # Iterate through each cutoff date
        for cutoff in cutoff_dates:
            # Create a copy of the original DataFrame to modify
            df_copy = df.copy()

            # Add the cutoff date as a new column
            df_copy['cutoff'] = cutoff

            # Define the sample column based on the cutoff date
            df_copy['sample'] = df_copy[date_col].apply(lambda x: 'test' if x == cutoff else ('test' if x > cutoff else 'train'))

            # Append the DataFrame with the new 'cutoff_date' and 'sample' columns to the results list
            results.append(df_copy)

        # Concatenate all the DataFrames into a single DataFrame
        final_df = pd.concat(results, ignore_index=True)

        return final_df

    # Add full horizon
    def add_horizon_last_cutoff(self, df, group_cols, date_col, horizon, freq='W'):
        """
        Ensure that for each group in the latest cutoff, we have the specified horizon (e.g., 13 weeks)
        after the cutoff date. If they don't exist, add them.
        All other data remains unchanged.

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

        # Sort the latest cutoff dataframe
        latest_df = latest_df.sort_values(by=group_cols + [date_col])

        # Function to process each group in the latest cutoff
        def process_group(group):
            # Get the cutoff date for this group
            cutoff_date = pd.to_datetime(group['cutoff'].iloc[0])

            # Generate the required date range starting from the cutoff date
            required_dates = pd.date_range(start=cutoff_date, periods=horizon + 1, freq=freq)[1:]

            # Create a DataFrame for the required dates
            new_dates_df = pd.DataFrame({date_col: required_dates})
            new_dates_df['sample'] = 'test'
            new_dates_df['cutoff'] = latest_cutoff

            # Add other group columns
            for col in group_cols:
                new_dates_df[col] = group[col].iloc[0]

            # Merge with the original group data
            merged = pd.concat([group, new_dates_df]).drop_duplicates(subset=group_cols + [date_col], keep='first')
            merged = merged.sort_values(by=date_col)

            return merged

        # Apply the processing function to each group in the latest cutoff
        processed_latest = latest_df.groupby(group_cols, group_keys=False).apply(process_group)

        # Combine the processed latest cutoff data with the rest of the data
        result = pd.concat([rest_df, processed_latest], ignore_index=True)

        # Fill forward all string columns
        string_columns = result.select_dtypes(include='object').columns
        result[string_columns] = result[string_columns].ffill()

        # Sort the final result
        result = result.sort_values(by=group_cols + ['cutoff', date_col])

        return result