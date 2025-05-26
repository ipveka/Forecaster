#!/bin/bash
# Script to run the forecaster with proper Python path

# Go to project root
cd "$(dirname "$0")"

# Set PYTHONPATH to include the project root
export PYTHONPATH="$PWD"

# Run the script with args
python src/run_forecaster.py "$@"
