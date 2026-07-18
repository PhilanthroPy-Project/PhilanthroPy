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
