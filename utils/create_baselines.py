# General libraries
import gc
import os
import warnings
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import psutil
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
        # Start function
        print("Starting baseline creation...")

        # Get frequency-specific parameters
        from utils.forecaster_utils import get_frequency_params

        freq_params = get_frequency_params(freq)

        # Use frequency-specific parameters if not provided by user
        if bs_window_size is None:
            bs_window_size = freq_params["bs_window_size"]
            print(f"Using frequency-based window size for baselines: {bs_window_size}")

        # Make a copy of the input DataFrame to avoid modifying the original
        result_df = df.copy()

        # Check if feature_cols is provided when LR or ML baselines are requested
        if ("LR" in baseline_types or "ML" in baseline_types) and feature_cols is None:
            raise ValueError(
                "feature_cols must be provided for 'LR' or 'ML' baseline types"
            )

        # Create baselines based on user preferences
        for baseline_type in baseline_types:
            if baseline_type == "MA":
                print("Creating Moving Average (MA) baselines...")
                result_df = self.create_ma_baseline(
                    result_df, group_cols, date_col, signal_cols, bs_window_size
                )
                print("MA baselines created successfully.")

            elif baseline_type == "LR":
                print("Creating Linear Regression (LR) baselines...")
                result_df = self.create_lr_baseline(
                    result_df, group_cols, date_col, signal_cols, feature_cols
                )
                print("LR baselines created successfully.")

            elif baseline_type == "ML":
                print("Creating LightGBM (ML) baselines...")
                result_df = self.create_lgbm_baseline(
                    result_df, group_cols, date_col, signal_cols, feature_cols
                )
                print("ML baselines created successfully.")

            else:
                print(f"Warning: Unknown baseline type '{baseline_type}'. Skipping.")

        print("Baseline creation completed.")
        return result_df

    # MA Baseline
    def create_ma_baseline(self, df, group_cols, date_col, signal_cols, bs_window_size):
        """
        Add moving average (MA) baselines and feature baselines for each signal column to the test set.

        Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'client', 'warehouse', 'product', 'date', 'sales', 'price',
                           'filled_sales', 'filled_price', and 'sample'.
        group_cols (list): The list of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        signal_cols (list): The list of signal columns to use for calculating the baselines.
        bs_window_size (int): The size of the moving average window.

        Returns:
        pd.DataFrame: A DataFrame with additional columns: 'baseline_{signal_col}' and 'feature_baseline_{signal_col}'
                      for each signal column.
        """
        # Sort the DataFrame by the group and date columns for consistent ordering
        df = df.sort_values(by=group_cols + [date_col])

        # Split the data into training and test sets
        train_df = df[df["sample"] == "train"]
        test_df = df[df["sample"] == "test"]

        # Iterate over each signal column to calculate the moving average baseline
        for signal_col in signal_cols:
            # Calculate the rolling mean (moving average) for each group and signal column
            train_df[
                f"feature_baseline_{signal_col}_ma_{bs_window_size}"
            ] = train_df.groupby(group_cols)[signal_col].transform(
                lambda x: x.rolling(bs_window_size, min_periods=1).mean()
            )

            # Extract the last moving average value from the training data for each group to apply to the test set
            last_ma_values = (
                train_df.groupby(group_cols)[
                    f"feature_baseline_{signal_col}_ma_{bs_window_size}"
                ]
                .last()
                .reset_index()
            )

            # Rename the column to reflect that this is the final baseline for the signal with MA method and window size
            last_ma_values = last_ma_values.rename(
                columns={
                    f"feature_baseline_{signal_col}_ma_{bs_window_size}": f"baseline_{signal_col}_ma_{bs_window_size}"
                }
            )

            # Merge the calculated baseline with the test set
            test_df = test_df.merge(last_ma_values, on=group_cols, how="left")

            # Set the feature baseline for the test set to the calculated baseline values
            test_df[f"feature_baseline_{signal_col}_ma_{bs_window_size}"] = test_df[
                f"baseline_{signal_col}_ma_{bs_window_size}"
            ]

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
