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

# Feature engineering class

class FeatureEngineering:
    def __init__(self):
        pass

    # Prepare data
    def run_feature_engineering(self, df, group_cols, date_col, target, horizon, freq,
                                window_sizes=(4, 13), lags=(4, 13), n_clusters=10,
                                train_weight_type='linear'):
        """
        Main function to prepare the data by calling all internal functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        group_cols (list): Columns to group the data
        date_col (str): Column containing dates
        target (str): Target variable
        horizon (int): Forecasting horizon
        freq (str): Frequency of the data
        window_sizes (tuple): Window sizes for stats
        lags (tuple): Lag values for creating lag features
        n_clusters (int): Number of groups for quantile clustering
        train_weight_type (str): Type of weighting for train weights

        Returns:
        pd.DataFrame: Prepared DataFrame
        """

        # Find signal and categorical columns
        signal_cols = [col for col in df.select_dtypes(include=['float64']).columns if col != target]

        # Get categorical columns for encoding
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        categorical_columns = [col for col in categorical_columns if col != 'sample']

        # Create encoded features
        df = self.create_encoded_features(df, categorical_columns)

        # Add date features
        df = self.create_date_features(df, date_col, freq)

        # Add periods feature
        df = self.create_periods_feature(df, target)

        # Add MA features
        df = self.create_ma_features(df, group_cols, signal_cols, window_sizes)

        # Add moving stats
        df = self.create_moving_stats(df, group_cols, signal_cols, window_sizes)

        # Add lag features if any feature columns are found
        signal_cols = [col for col in df.select_dtypes(include=['float64']).columns if col != target]
        df = self.create_lag_features(df, group_cols, date_col, signal_cols, lags, horizon)

        # Add coefficient of variance for target
        df = self.create_cov(df, group_cols, target)

        # Add quantile clusters
        df = self.create_quantile_clusters(df, group_cols, target, n_clusters)

        # Add history clusters
        df = self.create_history_clusters(df, group_cols, target, n_clusters)

        # Add train weights
        df = self.create_train_weights(df, group_cols, train_weight_type)

        # Return
        return df

    # Create encoded features
    def create_encoded_features(self, df, categorical_columns):
        """
        Add label-encoded features for the specified categorical columns in the dataframe.

        Parameters:
        - df: pandas DataFrame
        - categorical_columns: list of column names to be label encoded

        Returns:
        - The modified DataFrame with new label-encoded columns prefixed with 'feature_'
        """

        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Label encode each specified column
        for col in categorical_columns:
            # Create a LabelEncoder and fit_transform the column, adding the transformed column with a 'feature_' prefix
            df_copy[f'feature_{col}'] = LabelEncoder().fit_transform(df_copy[col])

        return df_copy

    # Create periods feature
    def create_periods_feature(self, df, group_columns, date_column, target_col):
        """
        Create a new feature 'feature_periods' that counts the number of weeks since
        the first non-zero signal for each group, based on the row order.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by (e.g., client, warehouse, product)
        - date_column: the column containing dates (e.g., 'date')
        - target_col: the column used to start counting when its value is greater than 0 (e.g., 'sales')

        Returns:
        - pandas DataFrame with new columns:
            - 'feature_periods' counting periods since the first non-zero signal,
            - 'feature_periods_expanding': expanded version of the periods count,
            - 'feature_periods_sqrt': square root of the periods count.
        """

        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Ensure the date_column is in datetime format
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

        # Sort by group_columns and the date_column to ensure proper order
        df_copy = df_copy.sort_values(by=group_columns + [date_column])

        # Create a mask to indicate rows where the signal_col is greater than 0
        df_copy['signal_above_zero'] = df_copy[target_col] > 0

        # Group by the group_columns and create a cumulative sum of the signal_above_zero mask to
        # start counting periods only when the signal_col is greater than 0
        df_copy['first_nonzero_signal'] = df_copy.groupby(group_columns)['signal_above_zero'].cumsum() > 0

        # Count periods only where the signal has been greater than zero
        df_copy['feature_periods'] = df_copy.groupby(group_columns).cumcount() + 1
        df_copy['feature_periods'] = df_copy['feature_periods'].where(df_copy['first_nonzero_signal'], 0)

        # Convert 'feature_periods' to float64
        df_copy['feature_periods'] = df_copy['feature_periods'].astype('float64')

        # Add feature_periods_expanding and feature_periods_sqrt
        df_copy['feature_periods_expanding'] = df_copy['feature_periods'] ** 1.10
        df_copy['feature_periods_sqrt'] = np.sqrt(df_copy['feature_periods'])

        # Ensure all new columns are float64
        df_copy['feature_periods_expanding'] = df_copy['feature_periods_expanding'].astype('float64')
        df_copy['feature_periods_sqrt'] = df_copy['feature_periods_sqrt'].astype('float64')

        # Reset the index after sorting and adding the feature_periods column
        df_copy = df_copy.reset_index(drop=True)

        # Drop the temporary columns
        df_copy = df_copy.drop(columns=['signal_above_zero', 'first_nonzero_signal'])

        return df_copy

    # Create date features
    def create_date_features(self, df: pd.DataFrame, date_col: str, freq: str) -> pd.DataFrame:
        """
        Create date-related features from the date column, including weeks and months until the next end of quarter and end of year.
        All generated features are prefixed with 'feature_' for consistency.

        Parameters:
            df (pd.DataFrame): Input DataFrame with a date column.
            date_col (str): The name of the date column from which to extract the features.
            freq (str): Frequency type ('W' for weekly, 'M' for monthly).

        Returns:
            pd.DataFrame: DataFrame with new feature columns, all prefixed with 'feature_'.
        """
        # Create a copy to avoid modifying the original DataFrame
        df = df.copy()

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Create basic date-related features using vectorized operations
        df['feature_year'] = df[date_col].dt.year
        df['feature_quarter'] = df[date_col].dt.quarter
        df['feature_month'] = df[date_col].dt.month

        # Create weekly features if specified
        if freq == 'W':
            df['feature_week'] = df[date_col].dt.isocalendar().week
        elif freq != 'M':
            raise ValueError("Frequency must be either 'W' for weekly or 'M' for monthly.")

        # Calculate next quarter end dates using pandas built-in function
        df['next_quarter_end'] = df[date_col] + pd.offsets.QuarterEnd(0)
        mask = df[date_col].dt.is_quarter_end
        df.loc[mask, 'next_quarter_end'] = df.loc[mask, date_col] + pd.offsets.QuarterEnd(1)

        # Calculate next year end dates
        df['next_year_end'] = df[date_col].dt.year.astype(str) + '-12-31'
        df['next_year_end'] = pd.to_datetime(df['next_year_end'])
        mask = df[date_col].dt.is_year_end
        df.loc[mask, 'next_year_end'] += pd.DateOffset(years=1)

        # Calculate weeks until next end of quarter/year using vectorized operations
        df['feature_weeks_until_next_end_of_quarter'] = ((df['next_quarter_end'] - df[date_col]).dt.days / 7).astype(int)
        df['feature_weeks_until_end_of_year'] = ((df['next_year_end'] - df[date_col]).dt.days / 7).astype(int)

        # Calculate months until next end of quarter/year more accurately
        df['feature_months_until_next_end_of_quarter'] = (
            (df['next_quarter_end'].dt.year - df[date_col].dt.year) * 12 +
            df['next_quarter_end'].dt.month - df[date_col].dt.month
        )

        df['feature_months_until_end_of_year'] = (
            (df['next_year_end'].dt.year - df[date_col].dt.year) * 12 +
            df['next_year_end'].dt.month - df[date_col].dt.month
        )

        # Drop intermediate columns
        df = df.drop(['next_quarter_end', 'next_year_end'], axis=1)

        return df

    # Calculate MA features
    def create_ma_features(self, df, group_columns, signal_columns, window_sizes):
        """
        Calculate moving averages for given signal columns, grouped by group columns, for multiple window sizes.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by
        - signal_columns: list of columns to calculate moving averages on
        - window_sizes: list of integers, each representing a window size for the moving average

        Returns:
        - pandas DataFrame with new columns for moving averages for each window size
        """

        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Loop through each signal column
        for signal_column in signal_columns:
            # Loop through each window size
            for window_size in window_sizes:
                # Create a new column name for each window size
                ma_column_name = f'{signal_column}_ma_{window_size}'

                # Calculate the moving average for the current signal column and window size
                df_copy[ma_column_name] = df_copy.groupby(group_columns)[signal_column]\
                    .transform(lambda x: x.rolling(window=window_size, min_periods=1).mean())

        return df_copy

    # Calculate moving stats
    def create_moving_stats(self, df, group_columns, signal_columns, window_sizes):
        """
        Calculate moving minimum and maximum for given signal columns, grouped by group columns, for multiple window sizes.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by
        - signal_columns: list of columns to calculate moving min/max on
        - window_sizes: list of integers, each representing a window size for the moving min/max

        Returns:
        - pandas DataFrame with new columns for moving minimum and maximum for each window size
        """

        # Make a copy of the dataframe to avoid modifying the original
        df_copy = df.copy()

        # Loop through each signal column
        for signal_column in signal_columns:
            # Loop through each window size
            for window_size in window_sizes:
                # Create new column names for moving minimum and maximum
                min_column_name = f'{signal_column}_min_{window_size}'
                max_column_name = f'{signal_column}_max_{window_size}'

                # Calculate the moving minimum for the current signal column and window size
                df_copy[min_column_name] = df_copy.groupby(group_columns)[signal_column]\
                    .transform(lambda x: x.rolling(window=window_size, min_periods=1).min())

                # Calculate the moving maximum for the current signal column and window size
                df_copy[max_column_name] = df_copy.groupby(group_columns)[signal_column]\
                    .transform(lambda x: x.rolling(window=window_size, min_periods=1).max())

        return df_copy

    # Create lag features
    def create_lag_features(self, df, group_columns, date_col, signal_columns, lags, forecast_window):
        """
        Create lag features for signal columns within each group, ensuring no data leakage for future forecasts,
        and fill forward the last known value only for lag feature columns.

        Parameters:
        df (pd.DataFrame): Input DataFrame with signal columns and group columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse']).
        date_col (str): The name of the date column used for sorting within each group.
        signal_columns (list): List of signal columns for which to create lag features (e.g., ['filled_sales', 'filled_price']).
        lags (list): List of lag values to calculate (e.g., [1, 2, ..., 26]).
        forecast_window (int): The forecasting window for which to prevent leakage (e.g., 13 for 13 weeks).

        Returns:
        pd.DataFrame: DataFrame with additional columns for lag features.
        """
        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort the DataFrame by group columns and date column
        df = df.sort_values(by=group_columns + [date_col])

        # Create lag features for each signal column
        lag_feature_columns = []
        for signal_column in signal_columns:
            for lag in lags:
                # Define the new feature column name
                lag_feature_column = f'feature_{signal_column}_lag_{lag}'
                lag_feature_columns.append(lag_feature_column)

                # Create lag feature by shifting the signal column within each group
                df[lag_feature_column] = df.groupby(group_columns)[signal_column].shift(lag)

        # Create a row number within each group
        df['row_num'] = df.groupby(group_columns).cumcount()

        # Identify the rows within the forecast window
        forecast_rows = df['row_num'] >= (len(df) - forecast_window)

        # For future rows (within the forecasting window), ensure no data leakage by clearing lagged values
        df.loc[forecast_rows, lag_feature_columns] = None

        # Fill forward the last known value for each group, only for lag feature columns
        df[lag_feature_columns] = (
            df.groupby(group_columns)[lag_feature_columns]
            .ffill()
        )

        # Drop the temporary row_num column
        df = df.drop(columns=['row_num'])

        return df

    # Create coefficient of variance
    def create_cov(self, df, group_columns, value_columns):
        """
        Calculate the coefficient of variation (CV) for multiple value columns within each group,
        excluding zeros from the calculation and only using rows where 'sample' equals 'train'.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group and value columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list): List of value columns for which to calculate CV (e.g., ['sales', 'price']).

        Returns:
        pd.DataFrame: DataFrame with additional CV columns for each value column, joined back to the full DataFrame.
        """
        # Copy the input DataFrame
        result = df.copy()

        # Filter only for rows where sample is 'train'
        train_data = df[df['sample'] == 'train']

        for value_column in value_columns:
            # Exclude zero values and calculate the mean and standard deviation for each group using the 'train' sample only
            group_stats = train_data.groupby(group_columns)[value_column].agg(['mean', 'std']).reset_index()

            # Calculate the coefficient of variation (CV = std / mean), handle cases where mean = 0 to avoid division by zero
            cov_column = f'feature_{value_column}_cov'
            group_stats[cov_column] = group_stats['std'] / group_stats['mean']
            group_stats[cov_column] = group_stats[cov_column].fillna(0)

            # Drop unnecessary columns 'mean' and 'std' after calculation
            group_stats = group_stats.drop(columns=['mean', 'std'])

            # Merge the CV back into the original DataFrame (result), keeping all original rows
            result = result.merge(group_stats[group_columns + [cov_column]], on=group_columns, how='left')

        return result

    # Create combinations
    def create_distinct_combinations(self, df, lower_level_group, group_columns):
        """
        For each combination of the lower-level group (e.g., 'product') with other group columns
        (e.g., 'client', 'warehouse', 'cutoff'), calculate the number of distinct values.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group columns.
        lower_level_group (str): The lower-level group column (e.g., 'product').
        group_columns (list): List of higher-level group columns (e.g., ['client', 'warehouse', 'cutoff']).

        Returns:
        pd.DataFrame: DataFrame with new columns for each combination's distinct counts.
        """
        # Make a copy of the input DataFrame
        result = df.copy()

        # Loop by group columns
        for group_col in group_columns:
            # Skip the lower_level_group if it is in group_columns
            if group_col == lower_level_group:
                continue

            # Create a new column name for this combination
            new_column_name = f'feature_distinct_{lower_level_group}_{group_col}'

            # Drop duplicates based on the lower-level group and the current group column
            distinct_combinations = df.drop_duplicates(subset=[lower_level_group, group_col])

            # Count the number of distinct values for each lower-level group
            combination_counts = distinct_combinations.groupby(lower_level_group)[group_col].nunique().reset_index(name=new_column_name)

            # Ensure the count column is of integer type
            combination_counts[new_column_name] = combination_counts[new_column_name].astype(int)

            # Merge the distinct count back into the original DataFrame
            result = result.merge(combination_counts, on=lower_level_group, how='left')

        return result

    # Create quantile clusters
    def create_quantile_clusters(self, df, group_columns, value_columns, n_groups=10):
        """
        Create quantile clusters for multiple value columns based on their mean values within each group,
        excluding zeros from the calculation and only using rows where 'sample' equals 'train'.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group and value columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list): List of value columns for which to create clusters (e.g., ['sales', 'price']).
        n_groups (int): Number of quantile groups to create for each value column (default is 4 for quartiles).

        Returns:
        pd.DataFrame: DataFrame with additional cluster columns for each value column, joined back to the full DataFrame.
        """
        # Copy the input DataFrame
        result = df.copy()

        # Filter only for rows where sample is 'train'
        train_data = df[df['sample'] == 'train']

        for value_column in value_columns:
            # Exclude zero values and calculate the mean for each group using the 'train' sample only
            avg_values = train_data.groupby(group_columns)[value_column].mean().reset_index()

            # Create quantile clusters based on the mean values
            cluster_column = f'feature_{value_column}_cluster'

            # Use `pd.qcut` to create quantiles and convert them to integers
            avg_values[cluster_column] = pd.qcut(avg_values[value_column], q=n_groups, duplicates='drop', labels=False) + 1

            # Ensure that the cluster column is of integer type
            avg_values[cluster_column] = avg_values[cluster_column].fillna(0).astype(int)

            # Merge the clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(avg_values[group_columns + [cluster_column]], on=group_columns, how='left')

        return result

    # Create history clusters
    def create_history_clusters(self, df, group_columns, value_columns, n_groups=10):
        """
        Create quantile clusters for multiple value columns based on their max values within each group,
        excluding zeros from the calculation and only using rows where 'sample' equals 'train'.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group and value columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list): List of value columns for which to create clusters (e.g., ['sales', 'price']).
        n_groups (int): Number of quantile groups to create for each value column (default is 4 for quartiles).

        Returns:
        pd.DataFrame: DataFrame with additional cluster columns for each value column, joined back to the full DataFrame.
        """
        # Copy the input DataFrame
        result = df.copy()

        # Filter only for rows where sample is 'train'
        train_data = df[df['sample'] == 'train']

        for value_column in value_columns:
            # Exclude zero values and calculate the max for each group using the 'train' sample only
            max_values = train_data.groupby(group_columns)[value_column].max().reset_index()

            # Create quantile clusters based on the max values
            cluster_column = f'{value_column}_history_cluster'

            # Use `pd.qcut` to create quantiles and convert them to integers
            max_values[cluster_column] = pd.qcut(max_values[value_column], q=n_groups, duplicates='drop', labels=False) + 1

            # Ensure that the cluster column is of integer type
            max_values[cluster_column] = max_values[cluster_column].fillna(0).astype(int)

            # Merge the clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(max_values[group_columns + [cluster_column]], on=group_columns, how='left')

        return result

    # Create intermittence clusters
    def create_intermittence_clusters(self, df, group_columns, value_columns, n_groups=10):
        """
        Create quantile clusters for each group based on the intermittence of multiple value columns.
        Intermittence is defined as the number of zero values divided by the total
        number of dates, excluding leading zeros, for each value column.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group and value columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list): List of value columns where zeros are evaluated for intermittence.
        n_groups (int): Number of quantile groups to create for the intermittence (default is 10).

        Returns:
        pd.DataFrame: DataFrame with additional intermittence cluster columns for each value column,
                      joined back to the full DataFrame.
        """
        # Copy the input DataFrame
        result = df.copy()

        # Function to calculate intermittence for a given column
        def calculate_intermittence(group, column):
            # Remove leading zeros
            non_zero_start_idx = (group[column] != 0).idxmax()
            trimmed_group = group.loc[non_zero_start_idx:]

            # Calculate intermittence: proportion of zero values after leading zeros
            zero_dates = (trimmed_group[column] == 0).sum()
            total_dates = trimmed_group.shape[0]

            # If no dates or all leading zeros were trimmed, intermittence is 0
            return zero_dates / total_dates if total_dates > 0 else 0

        # Filter only for rows where sample is 'train'
        train_data = df[df['sample'] == 'train']

        for value_column in value_columns:
            # Calculate intermittence for each group for the current value column
            intermittence_values = train_data.groupby(group_columns).apply(lambda group: calculate_intermittence(group, value_column)).reset_index(name=f'intermittence_{value_column}')

            # Create quantile clusters based on the intermittence for the current value column
            intermittence_values[f'feature_intermittence_{value_column}_cluster'] = pd.qcut(
                intermittence_values[f'intermittence_{value_column}'], q=n_groups,
                duplicates='drop', labels=False) + 1

            # Ensure that the cluster column is of integer type
            intermittence_values[f'feature_intermittence_{value_column}_cluster'] = intermittence_values[f'feature_intermittence_{value_column}_cluster'].fillna(0).astype(int)

            # Merge the intermittence clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(intermittence_values[group_columns + [f'feature_intermittence_{value_column}_cluster']],
                                  on=group_columns, how='left')

        return result

    # Create train weights
    def create_train_weights(self, df, group_cols, feature_periods_col='feature_periods', train_weight_type='LINEAR'):
        """
        Creates a 'weight' column for each group in a DataFrame, giving more weight to recent observations.
        Weights are calculated only for rows where 'sample' is 'train'.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        group_cols (list): List of columns to group by.
        feature_periods_col (str): Column name for the feature periods (e.g., weeks since inception).
        train_weight_type (str): The type of weighting to use ('exponential' or 'linear'). Defaults to 'exponential'.

        Returns:
        pd.DataFrame: A copy of the input DataFrame with an added 'weight' column for 'train' samples.
        """
        def compute_weights(group):
            # Calculate the maximum period within the group
            max_period = group[feature_periods_col].max()

            if train_weight_type == 'exponential':
                # Exponential weights: more weight to recent periods
                return np.exp((group[feature_periods_col] - max_period) / max_period)
            elif train_weight_type == 'linear':
                # Linear weights: weight proportional to feature_periods
                return group[feature_periods_col] / max_period
            else:
                raise ValueError("Invalid train_weight_type. Choose either 'exponential' or 'linear'.")

        # Create a copy of the input DataFrame to avoid modifying the original
        df_copy = df.copy()

        # Create a boolean mask for rows where 'sample' is 'train'
        mask = df_copy['sample'] == 'train'

        # Calculate and assign weights for 'train' samples
        df_copy.loc[mask, 'train_weight'] = (
            df_copy[mask]
            .groupby(group_cols)[feature_periods_col]
            .transform(lambda x: compute_weights(pd.DataFrame({feature_periods_col: x})))
        )

        return df_copy