# Standard library imports
import warnings

# Third-party imports
import numpy as np
import pandas as pd

# ML library imports
from lightgbm import LGBMRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import (
    GridSearchCV,
    HalvingRandomSearchCV,
    RandomizedSearchCV,
    TimeSeriesSplit,
)

# Options
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


# Define estimator map
estimator_map = {
    "LGBM": LGBMRegressor(
        n_jobs=-1, objective="regression", random_state=42, verbose=-1
    ),
    "RF": RandomForestRegressor(random_state=42, verbose=-1),
    "GBM": GradientBoostingRegressor(random_state=42, verbose=-1),
    "ADA": AdaBoostRegressor(random_state=42),
    "LR": LinearRegression(),
}

# Define parameter dictionary
hyperparam_dictionary = {
    "LGBM": {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [500, 1000, 2000],
        "num_leaves": [15, 31, 64],
        "max_depth": [4, 8, 12],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "reg_alpha": [0.0, 0.1, 0.5],
        "reg_lambda": [0.0, 0.1, 0.5],
    },
    "RF": {
        "n_estimators": [100, 500, 1000, 1500],
        "max_depth": [8, 16, 32, 64, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", None],
        "criterion": ["squared_error", "absolute_error"],
        "bootstrap": [True, False],
        "max_samples": [0.5, 0.7, 0.9],
    },
    "GBM": {
        "n_estimators": [100, 300, 500, 1000],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [3, 4, 6, 8],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.6, 0.8, 1.0],
        "max_features": ["sqrt", "log2", None],
        "validation_fraction": [0.1, 0.2],
    },
    "ADA": {
        "n_estimators": [50, 100, 200, 500],
        "learning_rate": [0.01, 0.05, 0.1, 0.3],
        "algorithm": ["SAMME", "SAMME.R"],
        "random_state": [42],
    },
    "LR": {
        "alpha": [0.001, 0.01, 0.1, 1.0, 10, 100, 500],
        "fit_intercept": [True, False],
        "solver": ["svd", "cholesky", "lsqr", "sag"],
        "max_iter": [1000],
        "normalize": [True, False],
        "tol": [1e-4, 1e-3],
    },
}


def create_param_heuristics(X_train, model_type="LGBM"):
    """
    Create heuristic-based hyperparameters based on dataset characteristics.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    model_type : str
        Type of model ('LGBM', 'RF', 'GBM', 'ADA', 'LR')

    Returns:
    --------
    tuple: (model, heuristic_params)
        - model: Configured model with heuristic hyperparameters
        - heuristic_params: Dictionary of parameters used
    """
    # Get dataset characteristics
    n_samples = X_train.shape[0]
    n_features = X_train.shape[1]

    # Store characteristics for logging
    dataset_info = {"n_samples": n_samples, "n_features": n_features}

    # Create heuristic parameters based on model type
    if model_type == "LGBM":
        # LightGBM heuristics
        # 1. n_estimators: More data/features -> more trees needed
        if n_samples < 1000:
            n_estimators = 100
        elif n_samples < 10000:
            n_estimators = 500
        elif n_samples < 100000:
            n_estimators = 1000
        else:
            n_estimators = 1500

        # 2. learning_rate: Adjust based on n_estimators
        if n_estimators <= 100:
            learning_rate = 0.1
        elif n_estimators <= 500:
            learning_rate = 0.05
        else:
            learning_rate = 0.01

        # 3. max_depth: Adjust based on n_features
        if n_features < 10:
            max_depth = 5
        elif n_features < 50:
            max_depth = 7
        elif n_features < 100:
            max_depth = 9
        else:
            max_depth = 12

        # 4. num_leaves: Related to max_depth but with some constraints
        # Rule of thumb: num_leaves = 2^(max_depth) but capped for better generalization
        num_leaves = min(2**max_depth, 127)  # Cap at 127 to prevent overfitting

        # 5. min_data_in_leaf: More data -> can afford larger min_data_in_leaf
        if n_samples < 1000:
            min_data_in_leaf = 5
        elif n_samples < 10000:
            min_data_in_leaf = 10
        elif n_samples < 100000:
            min_data_in_leaf = 20
        else:
            min_data_in_leaf = 50

        # 6. feature_fraction: More features -> lower feature_fraction to prevent overfitting
        if n_features < 10:
            feature_fraction = 1.0
        elif n_features < 50:
            feature_fraction = 0.9
        elif n_features < 100:
            feature_fraction = 0.8
        else:
            feature_fraction = 0.7

        # 7. bagging_fraction: Similar logic to feature_fraction
        if n_samples < 1000:
            bagging_fraction = 1.0
        elif n_samples < 10000:
            bagging_fraction = 0.9
        elif n_samples < 100000:
            bagging_fraction = 0.8
        else:
            bagging_fraction = 0.7

        # 8. Regularization: More features -> stronger regularization
        if n_features < 20:
            reg_alpha = 0.0
            reg_lambda = 0.0
        elif n_features < 50:
            reg_alpha = 0.01
            reg_lambda = 0.01
        elif n_features < 100:
            reg_alpha = 0.03
            reg_lambda = 0.05
        else:
            reg_alpha = 0.05
            reg_lambda = 0.1

        # Create model with heuristic parameters
        model = LGBMRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_data_in_leaf,
            colsample_bytree=feature_fraction,
            subsample=bagging_fraction,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=-1,
            objective="regression",
            random_state=42,
            verbose=-1,
        )

        # Store parameters for logging
        heuristic_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "num_leaves": num_leaves,
            "min_child_samples": min_data_in_leaf,
            "colsample_bytree": feature_fraction,
            "subsample": bagging_fraction,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "dataset_info": dataset_info,
        }

    elif model_type == "RF":
        # Random Forest heuristics
        # Adjust n_estimators based on dataset size
        if n_samples < 1000:
            n_estimators = 100
        elif n_samples < 10000:
            n_estimators = 200
        else:
            n_estimators = 500

        # Adjust max_depth based on n_features
        if n_features < 10:
            max_depth = 10
        elif n_features < 50:
            max_depth = 20
        else:
            max_depth = 30

        # Adjust min_samples_split and min_samples_leaf based on dataset size
        if n_samples < 1000:
            min_samples_split = 2
            min_samples_leaf = 1
        elif n_samples < 10000:
            min_samples_split = 5
            min_samples_leaf = 2
        else:
            min_samples_split = 10
            min_samples_leaf = 4

        # Create model with heuristic parameters
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )

        # Store parameters for logging
        heuristic_params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "dataset_info": dataset_info,
        }

    elif model_type == "GBM":
        # Gradient Boosting heuristics
        # Adjust n_estimators based on dataset size
        if n_samples < 1000:
            n_estimators = 100
            learning_rate = 0.1
        elif n_samples < 10000:
            n_estimators = 200
            learning_rate = 0.05
        else:
            n_estimators = 500
            learning_rate = 0.02

        # Adjust max_depth based on n_features
        if n_features < 10:
            max_depth = 3
        elif n_features < 50:
            max_depth = 4
        else:
            max_depth = 5

        # Create model with heuristic parameters
        model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=42,
            verbose=-1,
        )

        # Store parameters for logging
        heuristic_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "dataset_info": dataset_info,
        }

    elif model_type == "ADA":
        # AdaBoost heuristics
        # Adjust n_estimators based on dataset size
        if n_samples < 1000:
            n_estimators = 50
            learning_rate = 1.0
        elif n_samples < 10000:
            n_estimators = 100
            learning_rate = 0.5
        else:
            n_estimators = 200
            learning_rate = 0.3

        # Create model with heuristic parameters
        model = AdaBoostRegressor(
            n_estimators=n_estimators, learning_rate=learning_rate, random_state=42
        )

        # Store parameters for logging
        heuristic_params = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "dataset_info": dataset_info,
        }

    else:  # Linear Regression or other models
        # For Linear Regression, no hyperparameters to tune
        model = LinearRegression()

        # Store parameters for logging
        heuristic_params = {"dataset_info": dataset_info}

    return model, heuristic_params


def tune_hyperparameters(
    X_train,
    y_train,
    model="LGBM",
    params=None,
    param_distributions=None,
    search_method="halving",
    n_splits=4,
    scoring="neg_root_mean_squared_error",
    n_iter=30,
    sample_weight=None,
):
    """
    Tune hyperparameters using the specified search method.

    Parameters:
    -----------
    X_train : pd.DataFrame or np.ndarray
        Training features
    y_train : pd.Series or np.ndarray
        Training target
    model : str
        Model type key (e.g., 'LGBM', 'RF', 'GBM')
    params : dict
        Dictionary mapping model names to base estimators
    param_distributions : dict
        Dictionary mapping model names to parameter distributions
    search_method : str
        Search method: 'grid', 'random', or 'halving'
    n_splits : int
        Number of splits for time series cross-validation
    scoring : str
        Scoring metric for optimization
    n_iter : int
        Number of iterations for random/halving search
    sample_weight : array-like, optional
        Sample weights for training

    Returns:
    --------
    tuple: (best_estimator, best_params)
        - best_estimator: Trained model with best parameters
        - best_params: Dictionary of best parameters found
    """
    # Set up cross-validation
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # Select the estimator and parameter distribution
    base_model = params[model]
    param_dist = param_distributions[model]

    # Print the selected base model and parameter distribution
    print("Selected Base Model:", base_model)
    print("Parameter Distribution:", param_dist)

    # Choose the search method
    if search_method == "grid":
        search = GridSearchCV(
            estimator=base_model,
            param_grid=param_dist,
            cv=tscv,
            scoring=scoring,
            error_score="raise",
            n_jobs=-1,
            verbose=1,
        )
    elif search_method == "random":
        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            n_iter=n_iter,
            cv=tscv,
            scoring=scoring,
            error_score="raise",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
    elif search_method == "halving":
        search = HalvingRandomSearchCV(
            estimator=base_model,
            param_distributions=param_dist,
            max_resources=3000,
            aggressive_elimination=False,
            return_train_score=False,
            refit=True,
            cv=tscv,
            factor=3,
            scoring=scoring,
            error_score="raise",
            n_jobs=-1,
            verbose=1,
            random_state=42,
        )
    else:
        raise ValueError(
            "Invalid search_method. Choose 'grid', 'random', or 'halving'."
        )

    # Perform the search, including sample weights if provided
    if sample_weight is not None:
        search.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        search.fit(X_train, y_train)

    return search.best_estimator_, search.best_params_
