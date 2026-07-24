# Handle missing wealth data

Third-party wealth vendors rarely match every record in a database. This guide shows you how to fill the gaps without leaking information from your test set into training.

## Using `WealthScreeningImputer`

`WealthScreeningImputer` computes its fill statistics on the training set and applies them unchanged to any set you transform later. That separation is what keeps the imputer leakage-safe.

### Available strategies
* `median`: Fill with the training-set median. Robust to outliers.
* `mean`: Fill with the training-set mean.
* `zero`: Treat missing data as zero wealth. Aggressive.

### Missingness indicators
Set `add_indicator=True` and the imputer appends a `<col>__was_missing` column for each imputed field. Keep this on when the absence itself carries information. For many vendors, **missing data is a signal** — the donor might be a patient who hasn't been screened yet, or someone new to the database.

```python
from philanthropy.preprocessing import WealthScreeningImputer

imputer = WealthScreeningImputer(wealth_cols=["net_worth", "real_estate"], strategy="median", add_indicator=True)
# X_out = imputer.fit_transform(X)
```

## Using `WealthScreeningImputerKNN`

When a missing value can be estimated from similar donors, reach for KNN imputation:

```python
from philanthropy.preprocessing import WealthScreeningImputerKNN

knn_imputer = WealthScreeningImputerKNN(strategy="knn", n_neighbors=5, add_indicator=True)
# X_out = knn_imputer.fit_transform(X)
```

## Creating percentiles

Fundraising data is heavily right-skewed. `WealthPercentileTransformer` converts raw dollars to 0-100 ranks:

```python
from philanthropy.preprocessing import WealthPercentileTransformer

percentiler = WealthPercentileTransformer(wealth_cols=["net_worth"])
# X_out = percentiler.fit_transform(X)
```
