Development & Testing
=====================

To ensure the reliability of donor analytics models, PhilanthroPy maintains a rigorous testing suite that mirrors our GitHub CI (Continuous Integration) environment. Contributors and maintainers should run these tests locally before proposing changes.

Prerequisites
-------------
Ensure you have the development dependencies installed:

.. code-block:: bash

    pip install -e ".[dev]"

Running Tests Locally
---------------------

PhilanthroPy uses ``pytest`` alongside ``hypothesis`` for property-based testing and ``pytest-cov`` for coverage analysis.

1. High-Level Unit Tests & Coverage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Run the core test suite and verify that code coverage remains above 85%:

.. code-block:: bash

    pytest tests/ --cov=philanthropy --cov-fail-under=85

2. Scikit-Learn API Compliance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verify that all transformers and estimators strictly adhere to the scikit-learn API (checked via ``check_estimator``):

.. code-block:: bash

    pytest tests/test_sklearn_compat.py -v

3. Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~~~~
Verify the mathematical robustness of transformers using randomized data generation (Hypothesis):

.. code-block:: bash

    pytest tests/test_transformers_property.py -v --hypothesis-seed=12345

4. Documentation & Doctests
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Verify that all code examples in the docstrings are functional and that the documentation site builds without warnings:

.. code-block:: bash

    # Run doctests
    pytest philanthropy --doctest-modules

    # Build HTML documentation (with warnings as errors)
    rm -rf docs/_build docs/auto_examples
    sphinx-build -b html docs docs/_build/html -W --keep-going

Why local tests might differ from CI
------------------------------------
If tests pass locally but fail in GitHub Actions:

* **Cache Mismatch**: ``Sphinx-Gallery`` can leave stale cache files. Always use ``rm -rf docs/_build docs/auto_examples`` for a clean build.
* **Env Strictness**: CI uses Scikit-Learn 1.8+ which is strict about mixed-type dtypes in DataFrames (e.g., mixing dates and floats). Ensure your transformers use ``dtype=object`` validation for raw CRM data.
* **Symlinks**: GitHub Pages deployments will fail if the ``_build/html`` directory contains symlinks.
