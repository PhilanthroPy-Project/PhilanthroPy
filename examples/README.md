# Examples

Runnable, copy-pasteable scripts. Each is a smoke test in
[`tests/test_examples.py`](../tests/test_examples.py), so they never drift from
the real API.

```bash
pip install philanthropy
python examples/quickstart.py
```

| Script | What it shows |
|---|---|
| [`quickstart.py`](quickstart.py) | Train `DonorPropensityModel` on synthetic donors, rank a held-out pool by 0–100 affinity score. |
| [`unischema_to_scores.py`](unischema_to_scores.py) | The ecosystem flow: a [UniSchema](https://github.com/PhilanthroPy-Project/UniSchema) `ConstituentEvent` stream → donor features → scores, no glue code. |
| [`method_reference_lalakiya2025.py`](method_reference_lalakiya2025.py) | Reference implementation of the method in [Lalakiya 2025 (ICCED)](https://doi.org/10.1109/ICCED68324.2025.11325064): RFM → classifier panel → permutation importance → correlation, on synthetic donor data. |

## Notebooks

Under [`notebooks/`](notebooks/), each with a Colab badge, executed end to end in CI on every push
(`pytest --nbmake examples/notebooks`, one leg of the `lint` job so an API change breaks the
notebook before it breaks a reader's copy-paste):

| Notebook | What it shows |
|---|---|
| [`01_quickstart_propensity.ipynb`](notebooks/01_quickstart_propensity.ipynb) | The README quickstart, plus a call list, a distribution plot, and permutation importance. |
| [`02_temporal_leakage.ipynb`](notebooks/02_temporal_leakage.ipynb) | Builds the same features two ways, as-of each year versus over the whole export, and measures the inflation. This is the library's central argument. |
| [`03_grateful_patient_pipeline.ipynb`](notebooks/03_grateful_patient_pipeline.ipynb) | The academic-medical-center path: encounters, an `as_of` cutoff, service-line weighting, the solicitation window, routed through a `ColumnTransformer`. |

`examples/quickstart.ipynb` is a redirect to `01_quickstart_propensity.ipynb`, kept for one release
so old links do not break.
