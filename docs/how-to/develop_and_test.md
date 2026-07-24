# Develop and Test PhilanthroPy

Run the test suite before you propose a change. These are the same checks that run in GitHub CI, so passing them locally means your work is ready to review. The suite guards the reliability of the donor analytics models.

## Prerequisites

Install the development dependencies:

```bash
pip install -e ".[dev]"
```

## Running Tests Locally

The suite uses `pytest`, with `hypothesis` for property-based testing and `pytest-cov` for coverage.

### 1. High-Level Unit Tests & Coverage

Run the core suite and confirm coverage stays above 85%:

```bash
pytest tests/ --cov=philanthropy --cov-fail-under=85
```

### 2. Scikit-Learn API Compliance

Confirm every transformer and estimator adheres to the scikit-learn API:

```bash
pytest tests/test_sklearn_compat.py -v
```

### 3. Property-Based Testing

Check the transformers against randomized data:

```bash
pytest tests/test_transformers_property.py -v --hypothesis-seed=12345
```

### 4. Running Doctests

Check that the code examples in docstrings still run:

```bash
pytest philanthropy --doctest-modules
```
