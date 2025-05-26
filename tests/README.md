# Forecaster Tests

This directory contains tests for the Forecaster project, organized into:

- `unit/`: Unit tests for individual components

## Running Tests

### Prerequisites

Make sure you have `pytest` installed:

```bash
pip install pytest
```

### Running All Tests

From the project root directory:

```bash
pytest
```

### Running Specific Tests

To run only the data preparation tests:

```bash
pytest tests/unit/test_data_preparation.py
```

To run a specific test:

```bash
pytest tests/unit/test_data_preparation.py::TestDataPreparation::test_run_data_preparation_default_params
```

### Generating Test Coverage Report

To generate a test coverage report (requires `pytest-cov`):

```bash
pip install pytest-cov
pytest --cov=utils
```

## Test Structure

- Each test class corresponds to a class in the project
- Each test method focuses on a specific function or scenario
- Fixtures are used to set up common test data
