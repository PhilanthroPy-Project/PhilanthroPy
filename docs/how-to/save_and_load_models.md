# Save and load models

Train an estimator once, then reuse it. This guide saves a fitted PhilanthroPy model to disk and reloads it, so gift officers and downstream jobs can score prospects without retraining.

Use `philanthropy.utils.save_model` and `load_model`. They wrap `joblib` — the serialization tool scikit-learn recommends — and write a **bundle**: the fitted model plus the ordered feature list, the target name, and the scikit-learn / PhilanthroPy versions that produced it. `load_model` checks those versions for you and warns on a mismatch. Plain `joblib.load` cannot.

## Save and reload a fitted estimator

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.models import DonorPropensityModel
from philanthropy.utils import save_model, load_model

df = generate_synthetic_donor_data(n_samples=1000, random_state=42)
feature_cols = ["total_gift_amount", "years_active", "event_attendance_count"]
X = df[feature_cols].to_numpy()
y = df["is_major_donor"].to_numpy()

model = DonorPropensityModel(n_estimators=200, random_state=0).fit(X, y)

save_model(model, "donor_propensity.joblib", features=feature_cols, target="is_major_donor")

bundle = load_model("donor_propensity.joblib")
print(sorted(bundle))
print(bundle["features"])
print(bundle["model"].predict_affinity_score(X[:5]).round(2))
```

`load_model` returns the whole bundle, not a bare estimator, so the scoring job never has to hard-code the column order — read it back from `bundle["features"]`:

```python
scoring_frame = df.head(5)
X_score = scoring_frame[bundle["features"]].to_numpy()
scores = bundle["model"].predict_affinity_score(X_score)
assert scores.shape == (5,)
```

## Persist a full pipeline, not just the estimator

A bundle can hold any fitted scikit-learn object, and it should hold the **entire** `Pipeline`. The pipeline captures your preprocessing alongside the model, so the transforms applied at scoring time match those applied at training time. Mismatched transforms are the single most common source of train/serve skew.

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("model", DonorPropensityModel(n_estimators=200, random_state=0)),
])
pipe.fit(X, y)

save_model(pipe, "pipeline.joblib", features=feature_cols, target="is_major_donor")

reloaded = load_model("pipeline.joblib")["model"]
print(reloaded.predict_proba(X[:5])[:, 1].round(3))
```

## Version compatibility

!!! warning "Unpickling across scikit-learn versions is unsafe"
    scikit-learn does **not** guarantee that a model pickled under one version will load correctly under another. Loading an artifact built with a different scikit-learn version can raise an error, or — worse — load silently and produce wrong predictions. The same caution applies to the PhilanthroPy version.

`load_model` emits a `UserWarning` for each version that differs from the running environment. Treat those warnings as retrain triggers, not noise. To turn one into a hard failure:

```python
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("error", UserWarning)
    bundle = load_model("pipeline.joblib")   # raises if the versions moved
```

Pin both libraries exactly in the environment that trains *and* the environment that serves, so the warning never fires in the first place. Read the stored values back with `bundle["sklearn_version"]` and `bundle["philanthropy_version"]`.

## The CLI writes the same bundles

`philanthropy train` calls `save_model`, and `philanthropy score` / `validate` call `load_model`, so an artifact from the CLI loads in Python and vice versa. See [Use the CLI](use_the_cli.md).

!!! tip "Only unpickle artifacts you trust"
    `load_model` executes arbitrary code during unpickling, exactly like scikit-learn's own persisted estimators. Load model files only from sources you control — never from untrusted or user-supplied input.
