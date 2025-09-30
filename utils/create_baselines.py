# Standard library imports
import gc
import os
import warnings
from itertools import product

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil

# ML library imports
from lightgbm import LGBMRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
pd.options.mode.chained_assignment = None


# Create Baselines class
class CreateBaselines:
    # Init
    def __init__(self):
        """
        Initialize the CreateBaselines class. This class provides methods to calculate
        different types of baselines: moving average (MA), linear regression (LR),
        and LightGBM regression (LGBM).
        """
        pass

    # Main function to run baselines
    def run_baselines(
        self,
        df,
        group_cols,
        date_col,
        signal_cols,
        baseline_types=["MA"],
        bs_window_size=None,
        feature_cols=None,
        freq=None,
    ):
        """
        Main function to prepare baselines by calling specified baseline functions in order.

        Parameters:
        df (pd.DataFrame): Input DataFrame containing the data
        group_cols (list): Columns to group the data (e.g., ['client', 'warehouse', 'product'])
        date_col (str): Column containing dates
        signal_cols (list): List of signal columns to create baselines for
        baseline_types (list): List of baseline types to create ('MA' for moving average, 'LR' for linear regression, 'ML' for LightGBM)
        bs_window_size (int): Window size for moving average baseline (default: 13)
        feature_cols (list): Feature columns for regression models (required for 'LR' and 'ML' baseline types)

        Returns:
        pd.DataFrame: Prepared DataFrame with baseline columns added
        """
        # Print header
        print("\n" + "=" * 70)
        print("BASELINE CREATION")
        print("=" * 70)
        
        # Log initial dataset info
        initial_rows = len(df)
        initial_cols = len(df.columns)
        n_groups = df[group_cols].drop_duplicates().shape[0] if isinstance(group_cols, list) else df[group_cols].nunique()
        print(f"\n📊 Input Dataset:")
        print(f"   • Rows: {initial_rows:,}")
        print(f"   • Columns: {initial_cols}")
        print(f"   • Groups: {n_groups:,}")
        print(f"   • Signal columns: {len(signal_cols)}")
        for signal in signal_cols:
            print(f"      - {signal}")

        # Get frequency-specific parameters
        from utils.forecaster_utils import get_frequency_params

        freq_params = get_frequency_params(freq)

        # Use frequency-specific parameters if not provided by user
        if bs_window_size is None:
            bs_window_size = freq_params["bs_window_size"]
            print(f"\n📅 Using frequency-based window size: {bs_window_size}")
        else:
            print(f"\n📅 Using specified window size: {bs_window_size}")

        # Display baseline configuration
        print(f"\n🎯 Baseline Configuration:")
        print(f"   • Baseline types: {', '.join(baseline_types)}")
        print(f"   • Total baselines to create: {len(baseline_types)}")
        
        # Check requirements for regression baselines
        regression_types = [bt for bt in baseline_types if bt in ["LR", "ML"]]
        if regression_types:
            if feature_cols is None:
                raise ValueError(
                    f"feature_cols must be provided for {', '.join(regression_types)} baseline types"
                )
            print(f"   • Features for regression models: {len(feature_cols)}")

        # Make a copy of the input DataFrame to avoid modifying the original
        result_df = df.copy()
        baselines_created = []

        # Create baselines based on user preferences
        for idx, baseline_type in enumerate(baseline_types, 1):
            print(f"\n[{idx}/{len(baseline_types)}] Creating {baseline_type} baseline(s)...")
            
            cols_before = len(result_df.columns)
            
            if baseline_type == "MA":
                print(f"   📉 Type: Moving Average")
                print(f"   • Window size: {bs_window_size}")
                print(f"   • Signals: {len(signal_cols)}")
                
                result_df = self.create_ma_baseline(
                    result_df, group_cols, date_col, signal_cols, bs_window_size
                )
                
                cols_added = len(result_df.columns) - cols_before
                print(f"   ✓ Created {cols_added} baseline column(s)")
                for signal in signal_cols:
                    baseline_col = f"baseline_{signal}_ma_{bs_window_size}"
                    baselines_created.append(baseline_col)
                    print(f"      - {baseline_col}")

            elif baseline_type == "LR":
                print(f"   📈 Type: Linear Regression")
                print(f"   • Features: {len(feature_cols)}")
                print(f"   • Signals: {len(signal_cols)}")
                print(f"   • Training per group...")
                
                result_df = self.create_lr_baseline(
                    result_df, group_cols, date_col, signal_cols, feature_cols
                )
                
                cols_added = len(result_df.columns) - cols_before
                print(f"   ✓ Created {cols_added} baseline column(s)")
                for signal in signal_cols:
                    baseline_col = f"baseline_{signal}_lr"
                    baselines_created.append(baseline_col)
                    print(f"      - {baseline_col}")

            elif baseline_type == "ML":
                print(f"   🤖 Type: LightGBM Regression")
                print(f"   • Features: {len(feature_cols)}")
                print(f"   • Signals: {len(signal_cols)}")
                print(f"   • Training per group...")
                print(f"   • Model: n_estimators=1000, learning_rate=0.05")
                
                result_df = self.create_lgbm_baseline(
                    result_df, group_cols, date_col, signal_cols, feature_cols
                )
                
                cols_added = len(result_df.columns) - cols_before
                print(f"   ✓ Created {cols_added} baseline column(s)")
                for signal in signal_cols:
                    baseline_col = f"baseline_{signal}_lgbm"
                    baselines_created.append(baseline_col)
                    print(f"      - {baseline_col}")

            else:
                print(f"   ⚠️  Unknown baseline type '{baseline_type}' - skipping")

        # Final summary
        final_cols = len(result_df.columns)
        total_baselines = len(baselines_created)
        
        print(f"\n" + "=" * 70)
        print(f"✅ BASELINE CREATION COMPLETED")
        print(f"=" * 70)
        print(f"   • Final shape: {result_df.shape[0]:,} rows × {result_df.shape[1]} columns")
        print(f"   • Total baselines created: {total_baselines}")
        print(f"   • Baseline types: {', '.join(baseline_types)}")
        print(f"   • Window size: {bs_window_size}")
        print("=" * 70 + "\n")
        
        return result_df

    # MA Baseline
    def create_ma_baseline(self, df, group_cols, date_col, signal_cols, bs_window_size, create_features=False):
        """
        Add moving average (MA) baselines for each signal column, and optionally feature baselines.
        Supports one or multiple window sizes.

        Parameters:
        -----------
        df : pd.DataFrame
            The input DataFrame with columns like 'client', 'warehouse', 'product', 'date', 'sales', 'price',
            'filled_sales', 'filled_price', and 'sample'.
        group_cols : list
            The list of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col : str
            The name of the date column.
        signal_cols : list
            The list of signal columns to use for calculating the baselines.
        bs_window_size : int or list[int]
            The size(s) of the moving average window(s).
        create_features : bool, optional (default=False)
            If True, also create `feature_baseline_*` columns.
            If False, only create `baseline_*` columns.

        Returns:
        --------
        pd.DataFrame
            A DataFrame with additional columns:
            - Always:  'baseline_{signal_col}_ma_{window}'
            - If True: 'feature_baseline_{signal_col}_ma_{window}'
        """
        # Ensure bs_window_size is a list
        if isinstance(bs_window_size, int):
            window_sizes = [bs_window_size]
        else:
            window_sizes = list(bs_window_size)

        # Sort the DataFrame by the group and date columns for consistent ordering
        df = df.sort_values(by=group_cols + [date_col])

        # Split the data into training and test sets
        train_df = df[df["sample"] == "train"].copy()
        test_df = df[df["sample"] == "test"].copy()

        # Iterate over each signal column and window size
        for signal_col in signal_cols:
            for window in window_sizes:
                # Compute rolling mean (moving average) for training set
                rolling_ma = (
                    train_df.groupby(group_cols)[signal_col]
                    .transform(lambda x: x.rolling(window, min_periods=1).mean())
                )

                # Column names
                baseline_col = f"baseline_{signal_col}_ma_{window}"
                
                # Add the actual moving average to training set baseline column
                train_df[baseline_col] = rolling_ma

                # Extract last MA value per group → baseline for test set
                last_ma_values = (
                    train_df.groupby(group_cols)[baseline_col]
                    .last()
                    .reset_index()
                )

                # Merge with test set
                test_df = test_df.merge(last_ma_values, on=group_cols, how="left")

                if create_features:
                    # Create feature_baseline_* for training (same as baseline for train)
                    feature_col = f"feature_baseline_{signal_col}_ma_{window}"
                    train_df[feature_col] = rolling_ma

                    # Copy baseline into feature_baseline_* for test set
                    test_df[feature_col] = test_df[baseline_col]

        # Concatenate the modified train and test sets back together
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        return final_df

    # LR Baseline
    def create_lr_baseline(
        self, df, group_cols, date_col, signal_cols, feature_cols, debug=False
    ):
        """
        Add linear regression (LR) baselines and feature baselines for each signal column to the test set.
        For the train set, store the actual values in the baseline columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'client', 'warehouse', 'product', 'date', 'sales', 'price',
                           'filled_sales', 'filled_price', and 'sample'.
        group_cols (list): The list of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        signal_cols (list): The list of signal columns to predict using linear regression.
        feature_cols (list): The list of feature columns to use for training the linear regression model.

        Returns:
        pd.DataFrame: A DataFrame with additional columns: 'baseline_{signal_col}_lr' and 'feature_baseline_{signal_col}_lr'
                      for each signal column.
        """
        # Sort the data by the group columns and the date column for proper ordering
        df = df.sort_values(by=group_cols + [date_col])

        # Split the data into training and test sets
        train_df = df[df["sample"] == "train"]
        test_df = df[df["sample"] == "test"]

        # Initialize the linear regression model
        lr_model = LinearRegression()

        # Iterate over each signal column to train separate linear models
        for signal_col in signal_cols:
            # print(f"Processing signal: {signal_col}")

            # For the training set, store the actual signal values in the baseline columns
            train_df[f"baseline_{signal_col}_lr"] = train_df[signal_col]
            train_df[f"feature_baseline_{signal_col}_lr"] = train_df[signal_col]

            # Track the total number of groups for progress updates
            total_groups = len(train_df.groupby(group_cols))
            baseline_preds = []  # Store test predictions for each group

            # Loop through each group and train the linear model individually
            for group_counter, (group_values, group_train) in enumerate(
                train_df.groupby(group_cols), 1
            ):
                # Ensure there is enough data in the group to train a model
                if len(group_train) > 1:
                    y_train = group_train[signal_col].fillna(
                        0
                    )  # Use the signal column as the target variable
                    X_train = group_train[
                        feature_cols
                    ]  # Use the feature columns as input features

                    # Fit the linear regression model on the training data
                    lr_model.fit(X_train, y_train)

                    # Extract the corresponding group from the test set
                    group_test = test_df[
                        (test_df[group_cols] == np.array(group_values)).all(axis=1)
                    ]

                    # If there is test data for this group, make predictions
                    if not group_test.empty:
                        X_test = group_test[feature_cols]
                        preds = lr_model.predict(X_test)

                        # Store the predictions in the test DataFrame for this signal
                        group_test[f"baseline_{signal_col}_lr"] = preds
                        group_test[f"feature_baseline_{signal_col}_lr"] = preds

                        # Append the group test results to the list of predictions
                        baseline_preds.append(group_test)

                # Print progress update for tracking large datasets
                if debug:
                    print(
                        f"Group {group_counter}/{total_groups} processed. Progress: {(group_counter / total_groups) * 100:.2f}%"
                    )

            # Concatenate all group test predictions back into the test set
            if baseline_preds:
                test_df = pd.concat(baseline_preds, ignore_index=True)

        # Concatenate the modified training and test sets
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        # print("LR Baseline creation completed.")
        return final_df

    # LGBM Baseline
    def create_lgbm_baseline(
        self, df, group_cols, date_col, signal_cols, feature_cols, debug=False
    ):
        """
        Add LightGBM regression baselines and feature baselines for each signal column to the test set.
        For the train set, store the actual values in the baseline columns.

        Parameters:
        df (pd.DataFrame): The input DataFrame
        group_cols (list): The list of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        signal_cols (list): The list of signal columns to predict using LightGBM.
        feature_cols (list): The list of feature columns to use for training the regression model.

        Returns:
        pd.DataFrame: A DataFrame with additional columns: 'baseline_{signal_col}_lgbm' and 'feature_baseline_{signal_col}_lgbm'
                      for each signal column.
        """
        # Sort the data for consistency
        df = df.sort_values(by=group_cols + [date_col])

        # Split the data into training and test sets
        train_df = df[df["sample"] == "train"]
        test_df = df[df["sample"] == "test"]

        # Iterate over each signal column to train separate LightGBM models
        for signal_col in signal_cols:
            # print(f"Processing signal: {signal_col}")

            # Track the total number of groups for progress updates
            total_groups = len(train_df.groupby(group_cols))
            baseline_preds = []  # Store test predictions for each group

            # Store the actual signal values for the train set
            train_df[f"baseline_{signal_col}_lgbm"] = train_df[signal_col]
            train_df[f"feature_baseline_{signal_col}_lgbm"] = train_df[signal_col]

            # Loop through each group to train the LightGBM model individually
            for group_counter, (group_values, group_train) in enumerate(
                train_df.groupby(group_cols), 1
            ):
                # Ensure there is enough data in the group to train a model
                if len(group_train) > 1:
                    y_train = group_train[signal_col].fillna(
                        0
                    )  # The target variable is the signal column
                    X_train = group_train[
                        feature_cols
                    ]  # The feature columns are the input variables

                    # Initialize and fit the LightGBM model
                    lgbm_model = LGBMRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        num_leaves=31,
                        random_state=42,
                        verbose=-1,
                    )
                    lgbm_model.fit(X_train, y_train)

                    # Extract the corresponding group from the test set
                    group_test = test_df[
                        (test_df[group_cols] == np.array(group_values)).all(axis=1)
                    ]

                    # If there is test data for this group, make predictions
                    if not group_test.empty:
                        X_test = group_test[feature_cols]
                        preds = lgbm_model.predict(X_test)

                        # Store the predictions in the test DataFrame
                        group_test[f"baseline_{signal_col}_lgbm"] = preds
                        group_test[f"feature_baseline_{signal_col}_lgbm"] = preds

                        # Append the group test results to the list of predictions
                        baseline_preds.append(group_test)

                # Optionally, print progress for large datasets
                if debug:
                    print(
                        f"Group {group_counter}/{total_groups} processed. Progress: {(group_counter / total_groups) * 100:.2f}%"
                    )

            # Concatenate all group test predictions back into the test set
            if baseline_preds:
                test_df = pd.concat(baseline_preds, ignore_index=True)

        # Concatenate the modified training and test sets
        final_df = pd.concat([train_df, test_df], ignore_index=True)

        # Optionally, print completion message
        # print("LGBM Baseline creation completed.")
        return final_df
