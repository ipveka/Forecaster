#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FastAPI wrapper for the Forecaster package.
This API allows users to run forecasting pipelines by sending data and parameters via HTTP requests.
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from dotenv import load_dotenv
# FastAPI and related libraries
from fastapi import Depends, FastAPI, HTTPException, Security, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

# Get API configuration from environment variables
API_HOST = os.getenv("FORECASTER_API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("FORECASTER_API_PORT", 8000))
API_KEY = os.getenv("FORECASTER_API_KEY", "")

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import Runner class from utils package
from utils.runner import Runner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create FastAPI app
app = FastAPI(
    title="Forecaster API",
    description="API for time series forecasting",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Key security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def get_api_key(api_key_header: str = Security(api_key_header)):
    """Validate API key from header."""
    if not API_KEY:
        # If no API key is set in environment, don't require authentication
        return True

    if api_key_header == API_KEY:
        return True

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid API Key",
        headers={"WWW-Authenticate": "ApiKey"},
    )


# Define input models
class ForecastRequest(BaseModel):
    """Request model for forecasting API."""

    # Data
    data: List[Dict[str, Any]] = Field(
        ..., description="Data in JSON format (list of records)"
    )

    # Required parameters
    date_col: str = Field(..., description="Name of the date column")
    group_cols: List[str] = Field(..., description="List of columns to group by")
    signal_cols: List[str] = Field(
        ..., description="List of signal columns to forecast"
    )

    # Data preparation parameters
    target: Optional[str] = Field(
        None, description="Target column (if None, uses the first signal column)"
    )
    horizon: int = Field(4, description="Forecasting horizon")
    freq: Optional[str] = Field(
        None, description="Frequency of the data ('D', 'W', 'M')"
    )
    n_cutoffs: int = Field(1, description="Number of cutoffs for backtesting")
    complete_dataframe: bool = Field(
        True, description="Whether to fill in missing dates"
    )
    smoothing: bool = Field(True, description="Whether to apply smoothing")
    dp_window_size: int = Field(13, description="Window size for smoothing")

    # Feature engineering parameters
    fe_window_size: Union[int, List[int]] = Field(
        (4, 13), description="Window sizes for feature engineering"
    )
    lags: Union[int, List[int]] = Field(
        (4, 13), description="Lag values for creating lag features"
    )
    fill_lags: bool = Field(False, description="Whether to fill forward lags")
    n_clusters: int = Field(10, description="Number of groups for quantile clustering")

    # Baseline parameters
    baseline_types: List[str] = Field(
        ["MA"], description="Types of baselines to create (MA, LR, ML, CROSTON)"
    )
    bs_window_size: int = Field(
        13, description="Window size for moving average baseline"
    )
    create_features: bool = Field(
        False, description="Whether to create feature_baseline_* columns"
    )

    # Forecaster parameters
    training_group: Optional[str] = Field(
        None, description="Column to use for training groups"
    )
    model: str = Field("LGBM", description="Model to use for forecasting")
    tune_hyperparameters: bool = Field(
        False, description="Whether to tune hyperparameters"
    )
    search_method: str = Field(
        "halving", description="Search method for hyperparameter tuning"
    )
    scoring: str = Field(
        "neg_mean_squared_log_error",
        description="Scoring metric for hyperparameter tuning",
    )
    n_best_features: int = Field(15, description="Number of best features to select")
    use_lags: bool = Field(True, description="Whether to include lag features")
    remove_outliers: bool = Field(
        False, description="Whether to remove outliers in forecasting"
    )
    outlier_column: Optional[str] = Field(
        None, description="Column from which to remove outliers"
    )
    lower_quantile: float = Field(
        0.025, description="Lower quantile for outlier removal"
    )
    upper_quantile: float = Field(
        0.975, description="Upper quantile for outlier removal"
    )
    ts_decomposition: bool = Field(
        False, description="Whether to use time series decomposition"
    )
    baseline_col: Optional[str] = Field(
        None, description="Baseline column to use for comparison"
    )
    use_guardrail: bool = Field(False, description="Whether to use guardrail limits")
    guardrail_limit: float = Field(
        2.5, description="Limit value for guardrail adjustments"
    )
    use_weights: bool = Field(False, description="Whether to use weights in training")
    use_parallel: bool = Field(
        True, description="Whether to run predictions in parallel"
    )
    num_cpus: Optional[int] = Field(
        None, description="Number of CPUs to use for parallel processing"
    )

    # Evaluator parameters
    eval_group_col: Optional[str] = Field(
        None, description="Column to group by for evaluation"
    )
    eval_group_filter: Optional[List[str]] = Field(
        None, description="Filter for groups in evaluation"
    )


class ForecastResponse(BaseModel):
    """Response model for forecasting API."""

    status: str
    execution_time: float
    data_shape: Dict[str, int]
    metrics: Optional[Dict[str, Any]]
    data: List[Dict[str, Any]]


@app.post("/forecast", response_model=ForecastResponse)
async def forecast(request: ForecastRequest, authorized: bool = Depends(get_api_key)):
    """
    Run a forecasting pipeline with the provided data and parameters.

    Returns the forecasting results including predictions, metrics, and feature importance.
    """
    try:
        # Convert input data to DataFrame
        df = pd.DataFrame(request.data)

        # Convert date column to datetime
        df[request.date_col] = pd.to_datetime(df[request.date_col])

        # Initialize Runner
        runner = Runner(verbose=True)

        # Convert window sizes and lags to tuples if they're lists
        fe_window_size = (
            tuple(request.fe_window_size)
            if isinstance(request.fe_window_size, list)
            else request.fe_window_size
        )
        lags = tuple(request.lags) if isinstance(request.lags, list) else request.lags

        # Start timer
        start_time = time.time()

        # Run the pipeline
        result_df = runner.run_pipeline(
            # Required parameters
            df=df,
            date_col=request.date_col,
            group_cols=request.group_cols,
            signal_cols=request.signal_cols,
            # Data preparation parameters
            target=request.target,
            horizon=request.horizon,
            freq=request.freq,
            n_cutoffs=request.n_cutoffs,
            complete_dataframe=request.complete_dataframe,
            smoothing=request.smoothing,
            dp_window_size=request.dp_window_size,
            # Feature engineering parameters
            fe_window_size=fe_window_size,
            lags=lags,
            fill_lags=request.fill_lags,
            n_clusters=request.n_clusters,
            # Baseline parameters
            baseline_types=request.baseline_types,
            bs_window_size=request.bs_window_size,
            create_features=request.create_features,
            # Forecaster parameters
            training_group=request.training_group,
            model=request.model,
            tune_hyperparameters=request.tune_hyperparameters,
            search_method=request.search_method,
            scoring=request.scoring,
            n_best_features=request.n_best_features,
            use_lags=request.use_lags,
            remove_outliers=request.remove_outliers,
            outlier_column=request.outlier_column,
            lower_quantile=request.lower_quantile,
            upper_quantile=request.upper_quantile,
            ts_decomposition=request.ts_decomposition,
            baseline_col=request.baseline_col,
            use_guardrail=request.use_guardrail,
            guardrail_limit=request.guardrail_limit,
            use_weights=request.use_weights,
            use_parallel=request.use_parallel,
            num_cpus=request.num_cpus,
            # Evaluator parameters
            eval_group_col=request.eval_group_col,
            eval_group_filter=request.eval_group_filter,
        )

        # Get execution time
        execution_time = time.time() - start_time

        # Get metrics
        metrics = runner.get_metrics()

        # Convert result DataFrame to records
        result_records = result_df.to_dict(orient="records")

        # Prepare response
        response = {
            "status": "success",
            "execution_time": execution_time,
            "data_shape": {"rows": result_df.shape[0], "columns": result_df.shape[1]},
            "metrics": metrics,
            "data": result_records,
        }

        return response

    except Exception as e:
        logging.error(f"Error in forecast endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check(authorized: bool = Depends(get_api_key)):
    """Health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
