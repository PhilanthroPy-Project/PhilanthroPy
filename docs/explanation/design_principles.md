# Design Principles

PhilanthroPy was built from the ground up to solve structural problems frequently found in non-profit data science.

* **Leakage-Safe by Design**: Temporal data leakage across fiscal years guarantees a model that performs flawlessly in backtests but fails in production. PhilanthroPy's transformers and `TemporalDonorSplitter` anchor cross-validation splits chronologically to the organization's fiscal-calendar to simulate realistic "walk-forward" predictions.
* **Idempotent Transformers**: Fill statistics, encounter summaries, and imputation snapshots are firmly frozen at `fit()` time. Calling `transform()` multiple times on streaming data will continuously yield identical, non-leaking transformations.
* **Scikit-Learn Native**: All estimators have been strictly tested against scikit-learn's `check_estimator` standard. They robustly support scikit-learn features like `set_output(transform="pandas")`, cloning, and cross-validation pipelines out of the box.
* **NaN-Transparent**: Real-world CRM data is fraught with empty and irregular fields. Third party database imports may miss up to 60% of their values. PhilanthroPy transforms operate with `allow_nan = True`, preventing silent data loss and correctly extracting signals from "missingness" itself.
* **PII-Aware**: Features like the `CRMCleaner` and `GratefulPatientFeaturizer` actively decouple clinical intensity and demographic logic from explicit Protected Health Information (PHI) such as Medical Record Numbers (MRNs) and Social Security Identifiers, minimizing compliance risks in analytical pipelines.
