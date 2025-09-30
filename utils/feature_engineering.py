# Standard library imports
import gc
import logging
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder
from statsmodels.tsa.seasonal import seasonal_decompose

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Pd options
pd.options.mode.chained_assignment = None


# Feature engineering class
class FeatureEngineering:
    # Init
    def __init__(self):
        pass

    # Prepare data
    def run_feature_engineering(
        self,
        df,
        group_cols,
        date_col,
        target,
        freq=None,
        fe_window_size=None,
        lags=None,
        fill_lags=False,
        n_clusters=10,
        extended_stats=False
    ):
        """
        Main function to prepare the data by calling all internal functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame
        group_cols (list): Columns to group the data
        date_col (str): Column containing dates
        target (str): Target variable
        freq (str): Frequency of the data
        fe_window_size (tuple): Window sizes for stats
        lags (tuple): Lag values for creating lag features
        fill_lags (bool): Whether to fill forward lags
        n_clusters (int): Number of groups for quantile clustering
        extended_stats (bool): Whether to add extended stats

        Returns:
        pd.DataFrame: Prepared DataFrame
        """
        # Print header
        print("\n" + "=" * 70)
        print("FEATURE ENGINEERING")
        print("=" * 70)
        
        # Log initial dataset info
        initial_rows = len(df)
        initial_cols = len(df.columns)
        initial_features = len([col for col in df.columns if col.startswith('feature_')])
        print(f"\n📊 Input Dataset:")
        print(f"   • Rows: {initial_rows:,}")
        print(f"   • Columns: {initial_cols}")
        print(f"   • Existing features: {initial_features}")
        print(f"   • Target: '{target}'")

        # Convert date column to datetime once at the beginning
        df[date_col] = pd.to_datetime(df[date_col])

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
        if fe_window_size is None:
            fe_window_size = freq_params["fe_window_size"]
            print(f"   ✓ Using frequency-based window sizes: {fe_window_size}")
        else:
            print(f"   ✓ Using specified window sizes: {fe_window_size}")

        if lags is None:
            lags = freq_params["lags"]
            print(f"   ✓ Using frequency-based lags: {lags}")
        else:
            print(f"   ✓ Using specified lags: {lags}")

        # Find categorical columns
        signal_cols = [col for col in df.select_dtypes(include=["float64"]).columns]
        print(f"\n📈 Identified {len(signal_cols)} signal column(s) for feature creation")

        # Get categorical columns for encoding
        categorical_columns = df.select_dtypes(include="object").columns.tolist()
        categorical_columns = [col for col in categorical_columns if col != "sample"]
        print(f"\n🏷️  Encoding {len(categorical_columns)} categorical column(s)...")
        if categorical_columns:
            for col in categorical_columns:
                n_unique = df[col].nunique()
                print(f"   • {col} ({n_unique} unique values)")
        
        # Create encoded features
        cols_before = len(df.columns)
        df = self.create_encoded_features(df, categorical_columns)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} encoded feature(s)")

        # Add date features
        print(f"\n📅 Creating temporal features...")
        cols_before = len(df.columns)
        df = self.create_date_features(df, date_col, extended_dates=False)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} date-based features")
        print(f"      - Basic: year, quarter, month, week, day, dayofweek")
        print(f"      - Cyclical: sin/cos encodings for seasonality")

        # Add periods feature
        print(f"\n⏱️  Creating period-based features...")
        cols_before = len(df.columns)
        df = self.create_periods_feature(df, group_cols, date_col, target)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} period feature(s)")
        print(f"      - Periods since first non-zero signal")

        # Find numeric columns, excluding cyclical features
        signal_cols = [
            col
            for col in df.select_dtypes(include=["float64"]).columns
            if (
                "feature_periods" not in col
                and "weeks_until" not in col
                and "months_until" not in col
                and "sin_" not in col
                and "cos_" not in col
                and "seasonal_" not in col
                and "trend_" not in col
            )
        ]
        print(f"\n📊 Processing {len(signal_cols)} signal(s) for MA/lag features")
        print(f"   ℹ️  Excluded cyclical/temporal features from calculations")

        # Add MA features
        print(f"\n📉 Creating moving average features...")
        cols_before = len(df.columns)
        df = self.create_ma_features(df, group_cols, signal_cols, fe_window_size)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} moving average feature(s)")
        print(f"      - Window sizes: {fe_window_size}")
        print(f"      - Signals: {len(signal_cols)}")

        # Add moving stats
        print(f"\n📊 Creating moving statistics features...")
        cols_before = len(df.columns)
        df = self.create_moving_stats(df, group_cols, signal_cols, fe_window_size)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} moving statistic feature(s)")
        print(f"      - Statistics: min, max, mean")
        print(f"      - Window sizes: {fe_window_size}")

        # Add lag features if any feature columns are found
        print(f"\n⏮️  Creating lag features...")
        cols_before = len(df.columns)
        df = self.create_lag_features(
            df, group_cols, date_col, signal_cols, lags, fill_lags
        )
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} lag feature(s)")
        print(f"      - Lag periods: {lags}")
        print(f"      - Fill forward: {fill_lags}")

        # Add coefficient of variance for target
        print(f"\n📐 Creating statistical features for target...")
        cols_before = len(df.columns)
        df = self.create_cov(df, group_cols, target)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created coefficient of variation feature")

        # Add quantile clusters
        print(f"\n🎯 Creating quantile cluster features...")
        cols_before = len(df.columns)
        df = self.create_quantile_clusters(df, group_cols, target, n_clusters)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} quantile cluster feature(s)")
        print(f"      - Number of clusters: {n_clusters}")

        # Add history clusters
        print(f"\n📚 Creating history cluster features...")
        cols_before = len(df.columns)
        df = self.create_history_clusters(df, group_cols, target, n_clusters)
        cols_added = len(df.columns) - cols_before
        print(f"   ✓ Created {cols_added} history cluster feature(s)")
        print(f"      - Number of clusters: {n_clusters}")

        # Extended stats
        if extended_stats:
            print(f"\n🔬 Creating extended statistical features...")
            cols_before = len(df.columns)
            
            # Add maximum consecutive zeros
            df = self.create_max_consecutive_zeros(df, group_cols, target)
            print(f"   ✓ Added maximum consecutive zeros feature")

            # Add Autocorrelation for target
            df = self.create_autocorr(df, group_cols, target, lag=1)
            print(f"   ✓ Added autocorrelation feature (lag=1)")
            
            cols_added = len(df.columns) - cols_before
            print(f"   ✓ Total extended features: {cols_added}")
        else:
            print(f"\n⏭️  Skipping extended statistics (extended_stats=False)")

        # Add train weights
        print(f"\n⚖️  Creating training weights...")
        df = self.create_train_weights(
            df,
            group_cols,
            feature_periods_col="feature_periods",
            train_weight_type="linear",
        )
        print(f"   ✓ Created 'train_weight' column (type: linear)")
        print(f"      - Gives more importance to recent observations")

        # Add forecast lag numbers
        print(f"\n🔢 Creating forecast lag numbers...")
        df = self.create_fcst_lag_number(df, group_cols, date_col)
        print(f"   ✓ Created 'fcst_lag_number' column")
        print(f"      - Tracks forecast horizon position")

        # Final summary
        final_cols = len(df.columns)
        final_features = len([col for col in df.columns if col.startswith('feature_')])
        features_added = final_features - initial_features
        
        print(f"\n" + "=" * 70)
        print(f"✅ FEATURE ENGINEERING COMPLETED")
        print(f"=" * 70)
        print(f"   • Final shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"   • Total features: {final_features} (+{features_added} new)")
        print(f"   • Frequency: {freq}")
        print(f"   • Window sizes: {fe_window_size}")
        print(f"   • Lags: {lags}")
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
            df_copy[f"feature_{col}"] = LabelEncoder().fit_transform(df_copy[col])

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
        # Copy the input DataFrame
        df_copy = df.copy()

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Ensure the date_column is in datetime format
        df_copy[date_column] = pd.to_datetime(df_copy[date_column])

        # Sort by group_columns and the date_column to ensure proper order
        df_copy = df_copy.sort_values(by=group_columns + [date_column])

        # Create a mask to indicate rows where the signal_col is greater than 0
        df_copy["signal_above_zero"] = df_copy[target_col] > 0

        # Group by the group_columns and create a cumulative sum of the signal_above_zero mask
        # Start counting periods only when the signal_col is greater than 0
        df_copy["first_nonzero_signal"] = (
            df_copy.groupby(group_columns)["signal_above_zero"].cumsum() > 0
        )

        # Count periods only where the signal has been greater than zero
        df_copy["feature_periods"] = df_copy.groupby(group_columns).cumcount() + 1
        df_copy["feature_periods"] = df_copy["feature_periods"].where(
            df_copy["first_nonzero_signal"], 0
        )

        # Convert 'feature_periods' to float64
        df_copy["feature_periods"] = df_copy["feature_periods"].astype("float64")

        # Add feature_periods_expanding and feature_periods_sqrt
        df_copy["feature_periods_expanding"] = df_copy["feature_periods"] ** 1.10
        df_copy["feature_periods_sqrt"] = np.sqrt(df_copy["feature_periods"])

        # Ensure all new columns are float64
        df_copy["feature_periods_expanding"] = df_copy[
            "feature_periods_expanding"
        ].astype("float64")
        df_copy["feature_periods_sqrt"] = df_copy["feature_periods_sqrt"].astype(
            "float64"
        )

        # Reset the index after sorting and adding the feature_periods column
        df_copy = df_copy.reset_index(drop=True)

        # Drop the temporary columns
        df_copy = df_copy.drop(columns=["signal_above_zero", "first_nonzero_signal"])

        return df_copy

    # Create date features
    def create_date_features(self, df, date_col, extended_dates=False):
        """
        Create date-related features from the date column, including weeks and months until the next end of quarter and end of year.
        All generated features are prefixed with 'feature_' for consistency.

        Parameters:
            df (pd.DataFrame): Input DataFrame with a date column.
            date_col (str): The name of the date column from which to extract the features.
            extended_dates (bool): Whether to include extended features

        Returns:
            pd.DataFrame: DataFrame with new feature columns, all prefixed with 'feature_'.
        """
        # Copy the input DataFrame
        df = df.copy()

        # Ensure the date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Create basic date-related features
        df["feature_year"] = df[date_col].dt.year
        df["feature_quarter"] = df[date_col].dt.quarter
        df["feature_month"] = df[date_col].dt.month

        # Lower frequency features
        df["feature_week"] = df[date_col].dt.isocalendar().week.astype(int)
        df["feature_day"] = df[date_col].dt.day
        df["feature_dayofweek"] = df[date_col].dt.dayofweek

        # Add sine and cosine features for cyclical patterns
        day_of_year = df[date_col].dt.dayofyear
        days_in_year = 365.25

        # Yearly seasonality - multiple harmonics (Fourier terms)
        for k in range(1, 4):  # 3 harmonics for yearly pattern
            df[f"feature_sin_yearly_{k}"] = np.sin(
                2 * np.pi * k * day_of_year / days_in_year
            )
            df[f"feature_cos_yearly_{k}"] = np.cos(
                2 * np.pi * k * day_of_year / days_in_year
            )

        # Quarterly seasonality
        days_in_quarter = 365.25 / 4
        day_of_quarter = day_of_year - (df["feature_quarter"] - 1) * days_in_quarter
        for k in range(1, 3):  # 2 harmonics for quarterly pattern
            df[f"feature_sin_quarterly_{k}"] = np.sin(
                2 * np.pi * k * day_of_quarter / days_in_quarter
            )
            df[f"feature_cos_quarterly_{k}"] = np.cos(
                2 * np.pi * k * day_of_quarter / days_in_quarter
            )

        # Monthly seasonality
        day_of_month = df[date_col].dt.day
        days_in_month = df[date_col].dt.days_in_month
        for k in range(1, 3):  # 2 harmonics for monthly pattern
            df[f"feature_sin_monthly_{k}"] = np.sin(
                2 * np.pi * k * day_of_month / days_in_month
            )
            df[f"feature_cos_monthly_{k}"] = np.cos(
                2 * np.pi * k * day_of_month / days_in_month
            )

        # Weekly seasonality
        day_of_week = df[date_col].dt.dayofweek
        for k in range(1, 4):  # 3 harmonics for weekly pattern
            df[f"feature_sin_weekly_{k}"] = np.sin(2 * np.pi * k * day_of_week / 7)
            df[f"feature_cos_weekly_{k}"] = np.cos(2 * np.pi * k * day_of_week / 7)

        # Whether to include extended features
        if extended_dates:
            # Calculate end of month dates
            df["next_month_end"] = df[date_col] + pd.offsets.MonthEnd(0)
            mask = df[date_col].dt.is_month_end
            df.loc[mask, "next_month_end"] = df.loc[
                mask, date_col
            ] + pd.offsets.MonthEnd(1)

            # Calculate end of week dates (assuming week ends on Sunday)
            df["next_week_end"] = df[date_col] + pd.offsets.Week(weekday=6)
            mask = df[date_col].dt.dayofweek == 6  # Sunday
            df.loc[mask, "next_week_end"] = df.loc[mask, date_col] + pd.offsets.Week(
                weekday=6, n=1
            )

            # Calculate next quarter end dates using pandas built-in function
            df["next_quarter_end"] = df[date_col] + pd.offsets.QuarterEnd(0)
            mask = df[date_col].dt.is_quarter_end
            df.loc[mask, "next_quarter_end"] = df.loc[
                mask, date_col
            ] + pd.offsets.QuarterEnd(1)

            # Calculate next year end dates
            df["next_year_end"] = df[date_col].dt.year.astype(str) + "-12-31"
            df["next_year_end"] = pd.to_datetime(df["next_year_end"])
            mask = df[date_col].dt.is_year_end
            df.loc[mask, "next_year_end"] += pd.DateOffset(years=1)

            # Calculate days until end of month/week/quarter/year as integers
            df["feature_days_until_end_of_month"] = (
                df["next_month_end"] - df[date_col]
            ).dt.days.astype(int)
            df["feature_days_until_end_of_week"] = (
                df["next_week_end"] - df[date_col]
            ).dt.days.astype(int)
            df["feature_days_until_end_of_quarter"] = (
                df["next_quarter_end"] - df[date_col]
            ).dt.days.astype(int)
            df["feature_days_until_end_of_year"] = (
                df["next_year_end"] - df[date_col]
            ).dt.days.astype(int)

            # Calculate weeks until end of month/quarter/year as floats
            df["feature_weeks_until_end_of_month"] = (
                df["feature_days_until_end_of_month"] / 7
            ).astype(float)
            df["feature_weeks_until_end_of_quarter"] = (
                df["feature_days_until_end_of_quarter"] / 7
            ).astype(float)
            df["feature_weeks_until_end_of_year"] = (
                df["feature_days_until_end_of_year"] / 7
            ).astype(float)

            # Calculate months until end of quarter/year as floats
            df["feature_months_until_end_of_quarter"] = (
                (
                    (df["next_quarter_end"].dt.year - df[date_col].dt.year) * 12
                    + df["next_quarter_end"].dt.month
                    - df[date_col].dt.month
                )
            ).astype(float)
            df["feature_months_until_end_of_year"] = (
                (
                    (df["next_year_end"].dt.year - df[date_col].dt.year) * 12
                    + df["next_year_end"].dt.month
                    - df[date_col].dt.month
                )
            ).astype(float)

            # Drop intermediate columns
            df = df.drop(
                [
                    "next_month_end",
                    "next_week_end",
                    "next_quarter_end",
                    "next_year_end",
                ],
                axis=1,
            )

        return df

    # Create time series decomposition features
    def create_decomposition_features(
        self, df, date_col, group_cols, signal_cols, period=None
    ):
        """
        Create time series decomposition features using seasonal_decompose.
        Extracts trend and seasonality components as features.

        Parameters:
            df (pd.DataFrame): Input DataFrame.
            date_col (str): The name of the date column.
            group_cols (list): Columns to group by.
            signal_cols (list): Signal columns to decompose.
            period (int, optional): Period for seasonal decomposition. If None, will try to infer.

        Returns:
            pd.DataFrame: DataFrame with decomposition features added.
        """
        # Copy the input DataFrame
        df = df.copy()

        # Ensure date column is in datetime format
        df[date_col] = pd.to_datetime(df[date_col])

        # Sort by date
        df = df.sort_values(date_col)

        # Infer period if not provided
        if period is None:
            # Try to infer from data frequency
            # Daily -> 7, Weekly -> 52, Monthly -> 12
            date_diff = df[date_col].diff().median().days
            if date_diff == 1:  # Daily data
                period = 7  # Weekly seasonality
            elif date_diff == 7:  # Weekly data
                period = 52  # Yearly seasonality
            elif date_diff >= 28 and date_diff <= 31:  # Monthly data
                period = 12  # Yearly seasonality
            else:
                period = 7  # Default

        # Process each group separately
        for group_vals, group_df in df.groupby(group_cols):
            # Convert group_vals to tuple if it's not already
            if not isinstance(group_vals, tuple):
                group_vals = (group_vals,)

            # Create mask for this group
            mask = True
            for col, val in zip(group_cols, group_vals):
                mask = mask & (df[col] == val)

            # Process each signal column
            for col in signal_cols:
                # Get time series for this group and signal
                ts = group_df[col]

                # Skip if not enough data points
                if len(ts) < period * 2:
                    continue

                try:
                    # Decompose the time series
                    decomposition = seasonal_decompose(
                        ts, model="additive", period=period
                    )

                    # Add trend and seasonal components as features
                    df.loc[mask, f"feature_trend_{col}"] = decomposition.trend
                    df.loc[mask, f"feature_seasonal_{col}"] = decomposition.seasonal

                    # Fill NaN values that occur at the beginning/end of trend
                    df[f"feature_trend_{col}"] = (
                        df[f"feature_trend_{col}"]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                    df[f"feature_seasonal_{col}"] = (
                        df[f"feature_seasonal_{col}"]
                        .fillna(method="ffill")
                        .fillna(method="bfill")
                    )
                except Exception as e:
                    print(f"Error decomposing {col} for group {group_vals}: {e}")
                    continue

        return df

    # Calculate MA features
    def create_ma_features(self, df, group_columns, signal_columns, fe_window_size):
        """
        Calculate moving averages for given signal columns, grouped by group columns, for multiple window sizes.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by
        - signal_columns: list of columns to calculate moving averages on
        - fe_window_size: list of integers, each representing a window size for the moving average

        Returns:
        - pandas DataFrame with new columns for moving averages for each window size
        """
        # Avoid unnecessary copy of the entire dataframe
        df_copy = df.copy()

        # Standardize inputs
        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        signal_columns = (
            [signal_columns] if isinstance(signal_columns, str) else signal_columns
        )
        fe_window_size = (
            [fe_window_size]
            if isinstance(fe_window_size, (int, float))
            else fe_window_size
        )

        # Validate input types
        if not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")
        if not isinstance(signal_columns, list):
            raise ValueError("signal_columns must be a list or a string.")

        # For each signal column and window size combination, calculate the moving average
        # More efficient to use transform directly with a string method name rather than lambda
        for signal_column in signal_columns:
            # Group once per signal column to avoid redundant groupby operations
            grouped = df_copy.groupby(group_columns)[signal_column]
            for window_size in fe_window_size:
                ma_column_name = f"{signal_column}_ma_{window_size}"
                df_copy[ma_column_name] = grouped.transform(
                    lambda x: x.rolling(window=window_size, min_periods=1).mean()
                )

        return df_copy

    # Calculate moving stats
    def create_moving_stats(self, df, group_columns, signal_columns, fe_window_size):
        """
        Calculate moving statistics (min, max, mean) for given signal columns, grouped by group columns, for multiple window sizes.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by
        - signal_columns: list of columns to calculate moving stats on
        - fe_window_size: list of integers, each representing a window size

        Returns:
        - pandas DataFrame with new columns for each moving stat per window size
        """
        df_copy = df.copy()

        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        signal_columns = (
            [signal_columns] if isinstance(signal_columns, str) else signal_columns
        )
        fe_window_size = (
            [fe_window_size]
            if isinstance(fe_window_size, (int, float))
            else fe_window_size
        )

        if not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")
        if not isinstance(signal_columns, list):
            raise ValueError("signal_columns must be a list or a string.")

        for signal_column in signal_columns:
            grouped = df_copy.groupby(group_columns)[signal_column]
            for window_size in fe_window_size:
                # Define stat functions and suffixes
                stats = {
                    "min": lambda x: x.rolling(window=window_size, min_periods=1).min(),
                    "max": lambda x: x.rolling(window=window_size, min_periods=1).max(),
                    "mean": lambda x: x.rolling(
                        window=window_size, min_periods=1
                    ).mean()
                }

                for stat_name, stat_func in stats.items():
                    col_name = f"{signal_column}_{stat_name}_{window_size}"
                    df_copy[col_name] = grouped.transform(stat_func)

        return df_copy

    # Create Maximum Consecutive Zeros
    def create_max_consecutive_zeros(self, df, group_columns, value_columns):
        """
        Calculate the maximum number of consecutive zeros for given value columns within each group.

        Parameters:
        - df (pd.DataFrame): Input DataFrame.
        - group_columns (list or str): Columns to group by (e.g. ['client', 'warehouse', 'product']).
        - value_columns (list or str): Columns to calculate max consecutive zeros on.

        Returns:
        - pd.DataFrame: Original DataFrame with new feature columns added.
        """
        import numpy as np

        result = df.copy()

        # Standardize inputs
        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        value_columns = (
            [value_columns] if isinstance(value_columns, str) else value_columns
        )

        for value_column in value_columns:
            if value_column not in df.columns:
                print(f"Warning: Column '{value_column}' not found in the DataFrame.")
                continue

            # Define the feature column name
            max_zeros_col = f"feature_{value_column}_max_consec_zeros"

            # Group and compute
            def max_consec_zeros(arr):
                is_zero = (arr == 0).astype(int)
                # Find runs of zeros
                padded = np.pad(is_zero, (1, 1), constant_values=(0, 0))
                diff = np.diff(padded)
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                if len(starts) == 0:
                    return 0
                return np.max(ends - starts)

            # Aggregate and merge
            agg_df = (
                df.groupby(group_columns)[value_column]
                .agg(lambda x: max_consec_zeros(x.values))
                .reset_index()
            )
            agg_df.rename(columns={value_column: max_zeros_col}, inplace=True)

            # Merge back to result
            result = result.merge(agg_df, on=group_columns, how="left")

        return result

    # Create lag features
    def create_lag_features(
        self, df, group_columns, date_col, signal_columns, lags, fill_lags
    ):
        """
        Create lag features for signal columns within each group, ensuring no data leakage for future forecasts
        by setting lags within the forecast window to None and limiting lag depth within the test set.

        Parameters:
        - df: pandas DataFrame
        - group_columns: list of columns to group by
        - date_col: column containing dates
        - signal_columns: list of columns to calculate lags for
        - lags: tuple or list of integers representing lag periods
        - fill_lags: whether to fill forward lag values (default: False)

        Returns:
        - pandas DataFrame with added lag features
        """
        # Avoid unnecessary copy if possible
        df_copy = df.copy()

        # Standardize inputs
        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        signal_columns = (
            [signal_columns] if isinstance(signal_columns, str) else signal_columns
        )
        lags = [lags] if isinstance(lags, (int, float)) else lags

        # Ensure inputs are valid
        if not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")
        if not isinstance(signal_columns, list):
            raise ValueError("signal_columns must be a list or a string.")

        # Ensure the date column is in datetime format
        df_copy[date_col] = pd.to_datetime(df_copy[date_col])

        # Sort the DataFrame by group columns and date column for proper shifting
        df_copy = df_copy.sort_values(by=group_columns + [date_col])

        # Create all lag features more efficiently in a single pass per signal column
        lag_feature_columns = []
        for signal_column in signal_columns:
            # Group once per signal column
            grouped = df_copy.groupby(group_columns)[signal_column]

            # Loop by lag
            for lag in lags:
                lag_feature_column = f"feature_{signal_column}_lag_{lag}"
                lag_feature_columns.append(lag_feature_column)
                df_copy[lag_feature_column] = grouped.shift(lag)

        # First create a mask for test rows
        test_mask = df_copy["sample"] == "test"

        # Only process test data if there are test rows
        if test_mask.any():
            # Initialize row_num column only for test rows to save memory
            df_copy["row_num"] = np.nan

            # Use cumcount for row numbering instead of iterating through groups
            df_copy.loc[test_mask, "row_num"] = (
                df_copy[test_mask].groupby(group_columns).cumcount()
            )

            # Vectorized operation to remove lag values that exceed available history
            for lag in lags:
                for signal_column in signal_columns:
                    lag_feature_column = f"feature_{signal_column}_lag_{lag}"
                    # Apply condition all at once
                    invalid_lags_mask = test_mask & (df_copy["row_num"] < lag)
                    if invalid_lags_mask.any():
                        df_copy.loc[invalid_lags_mask, lag_feature_column] = None

            # Fill forward the last known value for lags if requested
            if fill_lags and lag_feature_columns:
                for group_name, group_df in df_copy.groupby(group_columns):
                    group_indices = group_df.index
                    if len(group_indices) > 0:
                        df_copy.loc[group_indices, lag_feature_columns] = df_copy.loc[
                            group_indices, lag_feature_columns
                        ].ffill()

            # Drop the temporary row_num column
            df_copy = df_copy.drop(columns=["row_num"])

        return df_copy

    # Create coefficient of variance
    def create_cov(self, df, group_columns, value_columns):
        """
        Calculate the coefficient of variation (CV) for multiple value columns within each group,
        excluding zeros from the calculation and only using rows where 'sample' equals 'train'.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group and value columns.
        group_columns (list): List of columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list or str): List of value columns for which to calculate CV (e.g., ['sales', 'price']).
                                    Can also be a single column name as a string (e.g., 'sales').

        Returns:
        pd.DataFrame: DataFrame with additional CV columns for each value column, joined back to the full DataFrame.
        """
        # Avoid unnecessary copy if we only add columns
        result = df.copy()

        # Standardize inputs
        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        value_columns = (
            [value_columns] if isinstance(value_columns, str) else value_columns
        )

        # Validate input types
        if not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")
        if not isinstance(value_columns, list):
            raise ValueError("value_columns must be a list or a string.")

        # Create a mask for train data - avoid creating a separate DataFrame
        train_mask = df["sample"] == "train"

        # Process all value columns at once where possible
        for value_column in value_columns:
            if value_column not in df.columns:
                print(f"Warning: Column '{value_column}' not found in the DataFrame.")
                continue

            # Create combined mask for non-zero values in train data
            non_zero_train_mask = train_mask & (df[value_column] != 0)

            if non_zero_train_mask.sum() == 0:  # No valid data
                # Create empty column with zeros
                cov_column = f"feature_{value_column}_cov"
                result[cov_column] = 0.0
                continue

            # Calculate mean and std directly
            stats_df = (
                df.loc[non_zero_train_mask]
                .groupby(group_columns)[value_column]
                .agg(["mean", "std"])
            )

            # Some groups might have only one value or constant values, resulting in NaN std
            # Replace NaN std with 0
            stats_df["std"] = stats_df["std"].fillna(0)

            # Calculate CV (coefficient of variation)
            cov_column = f"feature_{value_column}_cov"
            stats_df[cov_column] = stats_df["std"] / stats_df["mean"]

            # Handle division by zero or near-zero means
            stats_df[cov_column] = (
                stats_df[cov_column].fillna(0).replace([np.inf, -np.inf], 0)
            )

            # Keep only the CV column for merging
            stats_df = stats_df[[cov_column]]

            # Merge with original DataFrame using an efficient left join
            result = result.merge(
                stats_df, left_on=group_columns, right_index=True, how="left"
            )

            # Fill NaN values with 0 and ensure proper data type
            result[cov_column] = result[cov_column].fillna(0).astype("float64")

        return result

    # Create autocorrelation
    def create_autocorr(
        self, df: pd.DataFrame, group_columns: list, value_columns: list, lag: int = 1
    ) -> pd.DataFrame:
        """
        Calculate lag-N autocorrelation for each value column within groups.

        Parameters:
        df (pd.DataFrame): Input DataFrame with time series data.
        group_columns (list or str): Columns to group by (e.g., ['client', 'warehouse', 'product']).
        value_columns (list or str): Target value columns to compute autocorrelation for (e.g., 'sales').
        lag (int): Lag at which to calculate autocorrelation (default is 1).

        Returns:
        pd.DataFrame: Original DataFrame with additional columns: 'feature_<col>_autocorr_lag<lag>'
        """
        # Copy the dataframe for safety
        result = df.copy()

        # Standardize inputs
        group_columns = (
            [group_columns] if isinstance(group_columns, str) else group_columns
        )
        value_columns = (
            [value_columns] if isinstance(value_columns, str) else value_columns
        )

        if not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")
        if not isinstance(value_columns, list):
            raise ValueError("value_columns must be a list or a string.")

        # Loop through each value column
        for value_column in value_columns:
            if value_column not in df.columns:
                print(f"Warning: Column '{value_column}' not found.")
                continue

            autocorr_col = f"feature_{value_column}_autocorr_lag{lag}"

            # Grouped autocorrelation calculation with error handling
            def autocorr_func(x):
                try:
                    # Check for sufficient data and non-zero variance
                    if len(x) <= lag + 1:
                        return np.nan
                        
                    # Check for constant series (variance = 0)
                    if np.isclose(x.var(), 0, atol=1e-10):
                        return np.nan
                        
                    # Calculate autocorrelation with error handling
                    with np.errstate(divide='ignore', invalid='ignore'):
                        corr = x.autocorr(lag)
                        
                    # Handle any invalid results
                    if not np.isfinite(corr):
                        return np.nan
                        
                    return corr
                    
                except Exception as e:
                    # Catch any unexpected errors during calculation
                    print(f"Warning in autocorrelation calculation: {str(e)}")
                    return np.nan

            # Compute autocorrelation per group
            autocorr_values = (
                df[df[value_column].notnull()]
                .groupby(group_columns)[value_column]
                .apply(autocorr_func)
                .fillna(0)
                .astype("float64")
            )

            # Join back to original df
            result = result.merge(
                autocorr_values.rename(autocorr_col),
                left_on=group_columns,
                right_index=True,
                how="left",
            )

            result[autocorr_col] = result[autocorr_col].fillna(0).astype("float64")

        return result

    # Create combinations
    def create_distinct_combinations(self, df, group_columns, lower_level_group):
        """
        For each combination of the lower-level group (e.g., 'product_number') with other group columns
        (e.g., 'reporterhq_id'), calculate the number of distinct values.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the group columns.
        group_columns (str or list): List of higher-level group columns (e.g., ['reporterhq_id']) or a single string.
        lower_level_group (str): The lower-level group column (e.g., 'product_number').

        Returns:
        pd.DataFrame: DataFrame with new columns for each combination's distinct counts.
        """
        # Copy the input DataFrame
        result = df.copy()

        # Ensure group_columns is a list of strings
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list) or not all(
            isinstance(col, str) for col in group_columns
        ):
            raise ValueError(
                "group_columns must be a list of strings or a single string."
            )

        # Remove lower_level_group from group_columns if present
        group_columns = [col for col in group_columns if col != lower_level_group]

        # Loop through each remaining group column
        for group_col in group_columns:
            # Create a new column name for this combination
            cluster_column = f"feature_distinct_{lower_level_group}_{group_col}"

            # Drop duplicates based on the lower-level group and the current group column
            distinct_combinations = df.drop_duplicates(
                subset=[lower_level_group, group_col]
            )

            # Count the number of distinct values for each lower-level group
            combination_counts = (
                distinct_combinations.groupby(lower_level_group)[
                    group_col
                ]  # Ensure this is a single column, not a list
                .nunique()
                .reset_index(name=cluster_column)
            )

            # Ensure the count column is of integer type
            combination_counts[cluster_column] = combination_counts[
                cluster_column
            ].astype(int)

            # Merge the distinct count back into the original DataFrame on the lower level group
            result = result.merge(combination_counts, on=lower_level_group, how="left")

            # Ensure the cluster column in the result DataFrame is an integer type
            result[cluster_column] = result[cluster_column].fillna(0).astype(int)

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

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Ensure value_columns is a list
        if isinstance(value_columns, str):
            value_columns = [value_columns]
        elif not isinstance(value_columns, list):
            raise ValueError("value_columns must be a list or a string.")

        # Filter only for rows where sample is 'train'
        train_data = df[df["sample"] == "train"]

        for value_column in value_columns:
            # Exclude zero values and calculate the mean for each group using the 'train' sample only
            avg_values = (
                train_data.groupby(group_columns)[value_column].mean().reset_index()
            )

            # Create quantile clusters based on the mean values
            cluster_column = f"feature_{value_column}_cluster"

            # Use `pd.qcut` to create quantiles and convert them to integers
            avg_values[cluster_column] = (
                pd.qcut(
                    avg_values[value_column],
                    q=n_groups,
                    duplicates="drop",
                    labels=False,
                )
                + 1
            )

            # Ensure that the cluster column is of integer type
            avg_values[cluster_column] = (
                avg_values[cluster_column].fillna(0).astype(int)
            )

            # Merge the clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(
                avg_values[group_columns + [cluster_column]],
                on=group_columns,
                how="left",
            )

            # Ensure the cluster column in the result DataFrame is an integer type
            result[cluster_column] = result[cluster_column].fillna(0).astype(int)

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

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Ensure value_columns is a list
        if isinstance(value_columns, str):
            value_columns = [value_columns]
        elif not isinstance(value_columns, list):
            raise ValueError("value_columns must be a list or a string.")

        # Filter only for rows where sample is 'train'
        train_data = df[df["sample"] == "train"]

        for value_column in value_columns:
            # Exclude zero values and calculate the max for each group using the 'train' sample only
            max_values = (
                train_data.groupby(group_columns)[value_column].max().reset_index()
            )

            # Create quantile clusters based on the max values
            cluster_column = f"feature_{value_column}_history_cluster"

            # Use `pd.qcut` to create quantiles and convert them to integers
            max_values[cluster_column] = (
                pd.qcut(
                    max_values[value_column],
                    q=n_groups,
                    duplicates="drop",
                    labels=False,
                )
                + 1
            )

            # Ensure that the cluster column is of integer type
            max_values[cluster_column] = (
                max_values[cluster_column].fillna(0).astype(int)
            )

            # Merge the clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(
                max_values[group_columns + [cluster_column]],
                on=group_columns,
                how="left",
            )

            # Ensure the cluster column in the result DataFrame is an integer type
            result[cluster_column] = result[cluster_column].fillna(0).astype(int)

        return result

    # Create intermittence clusters
    def create_intermittence_clusters(
        self, df, group_columns, value_columns, n_groups=10
    ):
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

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Ensure value_columns is a list
        if isinstance(value_columns, str):
            value_columns = [value_columns]
        elif not isinstance(value_columns, list):
            raise ValueError("value_columns must be a list or a string.")

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
        train_data = df[df["sample"] == "train"]

        # Loop by value column
        for value_column in value_columns:
            # Calculate intermittence for each group for the current value column
            intermittence_values = (
                train_data.groupby(group_columns)
                .apply(lambda group: calculate_intermittence(group, value_column))
                .reset_index(name=f"intermittence_{value_column}")
            )

            # Create new name for intermittence cluster
            cluster_name = f"feature_intermittence_{value_column}_cluster"

            # Create quantile clusters based on the intermittence for the current value column
            intermittence_values[cluster_name] = (
                pd.qcut(
                    intermittence_values[f"intermittence_{value_column}"],
                    q=n_groups,
                    duplicates="drop",
                    labels=False,
                )
                + 1
            )

            # Ensure that the cluster column is of integer type
            intermittence_values[cluster_name] = (
                intermittence_values[cluster_name].fillna(0).astype(int)
            )

            # Merge the intermittence clusters back into the original DataFrame (result), keeping all original rows
            result = result.merge(
                intermittence_values[group_columns + [cluster_name]],
                on=group_columns,
                how="left",
            )

            # Ensure the cluster column in the result DataFrame is an integer type
            result[cluster_name] = result[cluster_name].fillna(0).astype(int)

        return result

    # Create train weights
    def create_train_weights(
        self,
        df,
        group_columns,
        feature_periods_col="feature_periods",
        train_weight_type="linear",
    ):
        """
        Creates a 'weight' column for each group in a DataFrame, giving more weight to recent observations.
        Weights are calculated only for rows where 'sample' is 'train'.

        Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        group_columns (list): List of columns to group by.
        feature_periods_col (str): Column name for the feature periods (e.g., weeks since inception).
        train_weight_type (str): The type of weighting to use ('exponential' or 'linear'). Defaults to 'linear'.

        Returns:
        pd.DataFrame: A copy of the input DataFrame with an added 'weight' column for 'train' samples.
        """
        # Copy the input DataFrame
        df_copy = df.copy()

        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Function to compute weights
        def compute_weights(group):
            # Calculate the maximum period within the group
            max_period = group[feature_periods_col].max()

            if train_weight_type == "exponential":
                # Exponential weights: more weight to recent periods
                weights = np.exp((group[feature_periods_col] - max_period) / max_period)
            elif train_weight_type == "linear":
                # Linear weights: weight proportional to feature_periods
                weights = group[feature_periods_col] / max_period
            else:
                raise ValueError(
                    "Invalid train_weight_type. Choose either 'exponential' or 'linear'."
                )

            return weights

        # Create a boolean mask for rows where 'sample' is 'train'
        mask = df_copy["sample"] == "train"

        # Calculate and assign weights for 'train' samples in a more efficient way
        train_data = df_copy[mask].copy()

        # Apply the compute_weights function to each group
        for group_name, group_df in train_data.groupby(group_columns):
            # Convert group_name to tuple if it's a single item
            if not isinstance(group_name, tuple):
                group_name = (group_name,)

            # Apply compute_weights to this group
            weights = compute_weights(group_df)

            # Create a boolean mask to identify these rows in the original dataframe
            group_mask = mask.copy()
            for i, col in enumerate(group_columns):
                group_mask &= df_copy[col] == group_name[i]

            # Assign the weights
            df_copy.loc[group_mask, "train_weight"] = weights

        return df_copy

    # Create forecast lag number
    def create_fcst_lag_number(
        self,
        df,
        group_columns,
        date_col="date",
        sample_col="sample",
        target_sample="test",
    ):
        """
        Adds a column `fcst_lag` to the DataFrame that counts rows from 1 for each occurrence of `sample = target_sample`
        within groups defined by `group_columns`. The count starts from the first occurrence of `sample = target_sample` in
        each group, and all prior rows are set to NaN.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing data to process.
        group_columns : list
            Columns to group by, e.g., ['client', 'warehouse', 'product', 'cutoff'].
        date_col : str, optional
            Column to order rows by within each group. Default is 'date'.
        sample_col : str, optional
            Column that identifies the target sample, e.g., 'sample'.
        target_sample : str, optional
            Value within `sample_col` to start counting from. Default is 'test'.

        Returns:
        --------
        pd.DataFrame
            The DataFrame with an added `fcst_lag` column as integer.
        """
        # Ensure group_columns is a list
        if isinstance(group_columns, str):
            group_columns = [group_columns]
        elif not isinstance(group_columns, list):
            raise ValueError("group_columns must be a list or a string.")

        # Sort DataFrame by group columns and date
        df = df.sort_values(by=group_columns + [date_col]).copy()

        # Auxiliary forecast lag function
        def apply_fcst_lag(group):
            start_index = group.index[group[sample_col] == target_sample]
            if len(start_index) == 0:
                group["fcst_lag"] = np.nan
            else:
                first_test_index = start_index[0]
                # Use range with integers for fcst_lag
                group.loc[first_test_index:, "fcst_lag"] = range(
                    1, len(group.loc[first_test_index:]) + 1
                )
                if first_test_index > group.index[0]:
                    group.loc[: first_test_index - 1, "fcst_lag"] = np.nan
            return group

        # Initialize fcst_lag column with NaN
        df["fcst_lag"] = np.nan

        # Process each group separately instead of using groupby.apply
        for group_name, group_df in df.groupby(group_columns):
            # Get indices of this group
            group_indices = group_df.index

            # Find start index for target_sample
            start_indices = group_df.index[group_df[sample_col] == target_sample]

            if len(start_indices) > 0:
                first_test_index = start_indices[0]
                # Calculate lag numbers for this group
                test_indices = group_df.loc[first_test_index:].index
                df.loc[test_indices, "fcst_lag"] = range(1, len(test_indices) + 1)

        # Cast to integer after filling NaN
        df["fcst_lag"] = df["fcst_lag"].fillna(0).astype(int)

        return df
