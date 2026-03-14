# Develop and Test PhilanthroPy

To ensure the reliability of donor analytics models, PhilanthroPy maintains a rigorous testing suite that mirrors our GitHub CI environment. Contributors and maintainers should run these tests locally before proposing changes.

## Prerequisites

Ensure you have the development dependencies installed:

```bash
pip install -e ".[dev]"
```

## Running Tests Locally

PhilanthroPy uses `pytest` alongside `hypothesis` for property-based testing and `pytest-cov` for coverage analysis.

### 1. High-Level Unit Tests & Coverage

Run the core test suite and verify that code coverage remains above 85%:

```bash
pytest tests/ --cov=philanthropy --cov-fail-under=85
```

### 2. Scikit-Learn API Compliance

Verify that all transformers and estimators strictly adhere to the scikit-learn API:

```bash
pytest tests/test_sklearn_compat.py -v
```

### 3. Property-Based Testing

Verify the mathematical robustness of transformers using randomized data generation:

```bash
pytest tests/test_transformers_property.py -v --hypothesis-seed=12345
```

### 4. Running Doctests

Verify that all code examples in docstrings are functional:

```bash
pytest philanthropy --doctest-modules
```
