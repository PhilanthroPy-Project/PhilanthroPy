---
description: "Explain why one donor scored high: explain_donor_scores swaps each feature for its training median and reports the signed change in that donor's score."
---

# Explain a donor's score

`donor_feature_importance` tells you which signals the model leans on across the whole file. A gift officer asking "why is this donor on my list?" needs the answer for that one donor. `explain_donor_scores` gives it, with no dependency beyond scikit-learn and pandas.

## How it works

For each donor and each feature, the feature is replaced with a typical value (the training median, or the most frequent value for a categorical column) and the donor is rescored:

```
contribution = score(donor) - score(donor with that feature set to typical)
```

A positive contribution means the donor's real value pushed the score up compared with a typical donor; a negative one pulled it down. The top `top_k` features per donor come back, largest absolute contribution first.

## Example

```python
from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.inspection import explain_donor_scores
from philanthropy.models import MajorGiftClassifier

cols = ["total_gift_amount", "years_active", "event_attendance_count"]
df = generate_synthetic_donor_data(n_samples=1000, random_state=0)
train, today = df.iloc[:800], df.iloc[800:]

model = MajorGiftClassifier(random_state=0).fit(train[cols], train["is_major_donor"])
reasons = explain_donor_scores(
    model, today[cols], "predict_affinity_score", baseline=train[cols], top_k=2
)
print(reasons.head())
# columns: row, rank, feature, value, baseline, contribution
```

`row` is the donor's index label in `today`, so the reasons join straight back to your prospect list.

## Choosing the method and the baseline

* `method` is the name of any per-row scoring method: `predict_proba` (the default), `predict_affinity_score`, `predict_lapse_score`, `predict_retention_score`, `predict_intent_score`, `decision_function`, or `predict` on the regressors (`AskAmountRecommender`, `ShareOfWalletRegressor`, `FinancialForecastModel`). For per-class output it explains the positive class of a binary model and each donor's predicted stage for `MovesManagementClassifier`; pass `class_index` to pick a class yourself.
* `baseline` should be the **training** features (medians are taken from it) or a 1-D sequence of reference values in column order. When it is omitted, an estimator that stores its own training medians (`FinancialForecastModel.fill_values_`) uses those; anything else raises and asks for one.
* `NaN` in the donor's row is explained like any value: the contribution is the effect of the field being missing rather than typical.
* Fitted pipelines work too, so a DataFrame with categorical columns encoded inside the pipeline is explained by its raw column names.

## Reading the result honestly

* It changes one feature at a time. Contributions are exact for additive models but do not sum to the score in general, and correlated features (largest gift and total giving, say) can share or mask credit.
* A donor whose value is already close to typical gets a contribution near zero, even on an important feature. For a low-scoring donor that sits on the same side of the decision as the typical donor, every contribution can be zero: the explanation answers "what makes this donor different from typical", which is most useful for the high scorers you are about to call.
* This is an explanation of the model, not of the donor. See [Compliance considerations](../explanation/compliance_considerations.md) before showing reasons to anyone outside the prospect research team.
