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

# Create Baselines class

class CreateBaselines:
    def __init__(self):
        """
        Initialize the CreateBaselines class. This class provides methods to calculate
        different types of baselines: moving average (MA), linear regression (LR),
        and LightGBM regression (LGBM).
        """
        pass

    # MA Baseline
    def create_ma_baseline(self, df, group_cols, date_col, signal_cols, window_size):
        """
        Add moving average (MA) baselines and feature baselines for each signal column to the test set.

        Parameters:
        df (pd.DataFrame): The input DataFrame with columns 'client', 'warehouse', 'product', 'date', 'sales', 'price',
                           'filled_sales', 'filled_price', and 'sample'.
        group_cols (list): The list of columns to group by (e.g., ['client', 'warehouse', 'product']).
        date_col (str): The name of the date column.
        signal_cols (list): The list of signal columns to use for calculating the baselines.
        window_size (int): The size of the moving average window.

        Returns:
        pd.DataFrame: A DataFrame with additional columns: 'baseline_{signal_col}' and 'feature_baseline_{signal_col}'
                      for each signal column.
        """
        # Sort the DataFrame by the group and date columns for consistent ordering
        df = df.sort_values(by=group_cols + [date_col])

        # Split the data into training and test sets
        train_df = df[df['sample'] == 'train']
        test_df = df[df['sample'] == 'test']

        # Iterate over each signal column to calculate the moving average baseline
        for signal_col in signal_cols:
            # Calculate the rolling mean (moving average) for each group and signal column
            train_df[f'feature_baseline_{signal_col}'] = train_df.groupby(group_cols)[signal_col].transform(
                lambda x: x.rolling(window_size, min_periods=1).mean())

            # Extract the last moving average value from the training data for each group to apply to the test set
            last_ma_values = train_df.groupby(group_cols)[f'feature_baseline_{signal_col}'].last().reset_index()

            # Rename the column to reflect that this is the final baseline for the signal
            last_ma_values = last_ma_values.rename(columns={f'feature_baseline_{signal_col}': f'baseline_{signal_col}'})

            # Merge the calculated baseline with the test set
            test_df = test_df.merge(last_ma_values, on=group_cols, how='left')

            # Set the feature baseline for the test set to the calculated baseline values
            test_df[f'feature_baseline_{signal_col}'] = test_df[f'baseline_{signal_col}']

        # Concatenate the modified train and test sets back together
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        return final_df

    # LR Baseline
    def create_lr_baseline(self, df, group_cols, date_col, signal_cols, feature_cols):
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
        train_df = df[df['sample'] == 'train']
        test_df = df[df['sample'] == 'test']

        # Initialize the linear regression model
        lr_model = LinearRegression()

        # Iterate over each signal column to train separate linear models
        for signal_col in signal_cols:
            # print(f"Processing signal: {signal_col}")

            # For the training set, store the actual signal values in the baseline columns
            train_df[f'baseline_{signal_col}_lr'] = train_df[signal_col]
            train_df[f'feature_baseline_{signal_col}_lr'] = train_df[signal_col]

            # Track the total number of groups for progress updates
            total_groups = len(train_df.groupby(group_cols))
            baseline_preds = []  # Store test predictions for each group

            # Loop through each group and train the linear model individually
            for group_counter, (group_values, group_train) in enumerate(train_df.groupby(group_cols), 1):
                # Ensure there is enough data in the group to train a model
                if len(group_train) > 1:
                    y_train = group_train[signal_col].fillna(0)  # Use the signal column as the target variable
                    X_train = group_train[feature_cols]  # Use the feature columns as input features

                    # Fit the linear regression model on the training data
                    lr_model.fit(X_train, y_train)

                    # Extract the corresponding group from the test set
                    group_test = test_df[(test_df[group_cols] == np.array(group_values)).all(axis=1)]

                    # If there is test data for this group, make predictions
                    if not group_test.empty:
                        X_test = group_test[feature_cols]
                        preds = lr_model.predict(X_test)

                        # Store the predictions in the test DataFrame for this signal
                        group_test[f'baseline_{signal_col}_lr'] = preds
                        group_test[f'feature_baseline_{signal_col}_lr'] = preds

                        # Append the group test results to the list of predictions
                        baseline_preds.append(group_test)

                # Print progress update for tracking large datasets
                # print(f"Group {group_counter}/{total_groups} processed. Progress: {(group_counter / total_groups) * 100:.2f}%")

            # Concatenate all group test predictions back into the test set
            if baseline_preds:
                test_df = pd.concat(baseline_preds, ignore_index=True)

        # Concatenate the modified training and test sets
        final_df = pd.concat([train_df, test_df], ignore_index=True)
        # print("LR Baseline creation completed.")
        return final_df

    # LGBM Baseline
    def create_lgbm_baseline(self, df, group_cols, date_col, signal_cols, feature_cols):
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
        train_df = df[df['sample'] == 'train']
        test_df = df[df['sample'] == 'test']

        # Iterate over each signal column to train separate LightGBM models
        for signal_col in signal_cols:
            # print(f"Processing signal: {signal_col}")

            # Track the total number of groups for progress updates
            total_groups = len(train_df.groupby(group_cols))
            baseline_preds = []  # Store test predictions for each group

            # Store the actual signal values for the train set
            train_df[f'baseline_{signal_col}_lgbm'] = train_df[signal_col]
            train_df[f'feature_baseline_{signal_col}_lgbm'] = train_df[signal_col]

            # Loop through each group to train the LightGBM model individually
            for group_counter, (group_values, group_train) in enumerate(train_df.groupby(group_cols), 1):
                # Ensure there is enough data in the group to train a model
                if len(group_train) > 1:
                    y_train = group_train[signal_col].fillna(0)  # The target variable is the signal column
                    X_train = group_train[feature_cols]  # The feature columns are the input variables

                    # Initialize and fit the LightGBM model
                    lgbm_model = LGBMRegressor(n_estimators=1000, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
                    lgbm_model.fit(X_train, y_train)

                    # Extract the corresponding group from the test set
                    group_test = test_df[(test_df[group_cols] == np.array(group_values)).all(axis=1)]

                    # If there is test data for this group, make predictions
                    if not group_test.empty:
                        X_test = group_test[feature_cols]
                        preds = lgbm_model.predict(X_test)

                        # Store the predictions in the test DataFrame
                        group_test[f'baseline_{signal_col}_lgbm'] = preds
                        group_test[f'feature_baseline_{signal_col}_lgbm'] = preds

                        # Append the group test results to the list of predictions
                        baseline_preds.append(group_test)

                # Optionally, print progress for large datasets
                # print(f"Group {group_counter}/{total_groups} processed. Progress: {(group_counter / total_groups) * 100:.2f}%")

            # Concatenate all group test predictions back into the test set
            if baseline_preds:
                test_df = pd.concat(baseline_preds, ignore_index=True)

        # Concatenate the modified training and test sets
        final_df = pd.concat([train_df, test_df], ignore_index=True)

        # Optionally, print completion message
        # print("LGBM Baseline creation completed.")
        return final_df