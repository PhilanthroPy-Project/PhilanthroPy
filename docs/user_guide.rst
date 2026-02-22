.. _user_guide:

==========
User Guide
==========

Welcome to the PhilanthroPy User Guide! This package is intended to provide scikit-learn compatible components specifically designed for predictive donor analytics.

Installation
============

You can install PhilanthroPy using pip:

.. code-block:: bash

    pip install philanthropy

Or with Conda:

.. code-block:: bash

    conda env create -f environment.yml && conda activate Philanthropy
    pip install -e ".[dev]"

Quick Start
===========

.. code-block:: python

    from philanthropy.datasets import generate_synthetic_donor_data
    from philanthropy.models import DonorPropensityModel

    df = generate_synthetic_donor_data(n_samples=500, random_state=42)
    X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
    y = df["is_major_donor"].to_numpy()

    model = DonorPropensityModel(n_estimators=200, random_state=0)
    model.fit(X, y)
    scores = model.predict_affinity_score(X)   # 0–100 affinity scale

Feature Guides
==============

PhilanthroPy is organized into three main functional areas:

.. toctree::
   :maxdepth: 2

   user_guide/preprocessing
   user_guide/models
   user_guide/metrics
   user_guide/development

Basic Usage
===========

PhilanthroPy transformers are designed to be dropped directly into scikit-learn `Pipeline` objects. 

.. code-block:: python

    from sklearn.pipeline import Pipeline
    from philanthropy.preprocessing import (
        FiscalYearTransformer, WealthScreeningImputer, SolicitationWindowTransformer
    )
    from philanthropy.models import DonorPropensityModel

    pipe = Pipeline([
        ("fy",      FiscalYearTransformer(date_col="gift_date")),
        ("wealth",  WealthScreeningImputer(wealth_cols=["estimated_net_worth"])),
        ("window",  SolicitationWindowTransformer()),
        ("model",   DonorPropensityModel(n_estimators=200, random_state=0)),
    ])
    pipe.fit(X_train, y_train)
    scores = pipe.predict_proba(X_test)[:, 1]

Check out the :doc:`Examples </auto_examples/index>` section for complete end-to-end machine learning pipelines.
