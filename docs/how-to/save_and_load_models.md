# Save and Load Models

Once you have trained a PhilanthroPy estimator, you will want to persist it so gift officers and downstream jobs can score prospects without retraining. This guide shows you how to save and reload a fitted model with `joblib` — the serialization tool scikit-learn recommends for estimators.

Because PhilanthroPy estimators are scikit-learn native, the same `joblib.dump` / `joblib.load` calls work whether you are persisting a single estimator or a full `sklearn.pipeline.Pipeline`.

## Persisting a Single Estimator

Any fitted estimator can be written to disk and reloaded as-is. The reloaded object keeps every learned attribute (`estimator_`, `classes_`, `n_features_in_`), so its predictions are identical to the original.

```python
import joblib
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel

df = generate_synthetic_donor_data(n_samples=1000, random_state=42)
X = df[["total_gift_amount", "years_active", "event_attendance_count"]].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=200, random_state=0).fit(X, y)

# Save the fitted estimator.
joblib.dump(model, "model.joblib")

# Reload it (e.g. in a scoring job) and score prospects.
model = joblib.load("model.joblib")
print(model.predict_affinity_score(X[:5]).round(2))
```

## Persisting a Full Pipeline

The recommendation is to persist the **entire** `Pipeline`, not just the final estimator. The pipeline captures your preprocessing (scaling, imputation, featurization) alongside the model, so the transforms applied at scoring time exactly match those applied at training time — the single most common source of train/serve skew.

## Version Compatibility

!!! warning "Unpickling across scikit-learn versions is unsafe"
    scikit-learn does **not** guarantee that a model pickled under one version will load correctly under another. Loading an artifact built with a different scikit-learn version can raise an error, or — worse — load silently and produce wrong predictions. The same caution applies to the PhilanthroPy version, since a model's internal structure can change between releases.

To keep artifacts reproducible:

* **Pin both libraries** in the environment that trains *and* the environment that serves. PhilanthroPy requires `scikit-learn>=1.6`, but for persistence you want an exact match, for example:

    ```text
    scikit-learn==1.8.0
    philanthropy==0.4.0
    ```

* **Store the versions alongside the artifact.** Bundle the fitted object and the versions that produced it in a single dictionary, then verify the versions on load. If they do not match, retrain rather than trusting the reloaded model.

## End-to-End Example

This example is fully self-contained — it builds its data from the synthetic generator, so it can run in CI without any external files. It trains a pipeline, dumps it with its versions, reloads it, checks the versions, and predicts.

```python
import joblib
import sklearn

import philanthropy
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 1. Build a training set from the synthetic generator (no external files).
df = generate_synthetic_donor_data(n_samples=1000, random_state=42)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()
y = df["is_major_donor"].to_numpy()

# 2. Train a full pipeline (preprocessing + estimator).
pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
])
pipe.fit(X, y)

# 3. Bundle the fitted pipeline with the versions that produced it.
artifact = {
    "pipeline": pipe,
    "feature_cols": feature_cols,
    "sklearn_version": sklearn.__version__,
    "philanthropy_version": philanthropy.__version__,
}
joblib.dump(artifact, "donor_propensity.joblib")

# 4. Reload in a fresh process, verify versions, and score.
artifact = joblib.load("donor_propensity.joblib")

if artifact["sklearn_version"] != sklearn.__version__:
    raise RuntimeError(
        f"Artifact built with scikit-learn {artifact['sklearn_version']}, "
        f"but {sklearn.__version__} is installed. Unpickling across "
        f"scikit-learn versions is not supported — retrain instead."
    )

model = artifact["pipeline"]
predictions = model.predict(X[:5])
probabilities = model.predict_proba(X[:5])[:, 1]
print("predictions:", predictions)
print("P(major donor):", probabilities.round(3))
```

!!! tip "Only unpickle artifacts you trust"
    `joblib.load` executes arbitrary code during unpickling. Load model files only from sources you control — never from untrusted or user-supplied input.
