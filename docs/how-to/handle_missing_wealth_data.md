# Handle Missing Wealth Data

Third-party wealth vendors rarely match 100% of a database. This guide shows you how to impute missing wealth information safely.

## Using `WealthScreeningImputer`

The `WealthScreeningImputer` handles the "missingness gap" safely by calculating statistics on the training set and applying them seamlessly to test sets, preventing data leakage.

### Available Strategies
* `median`: Fill with training-set median (Robust to outliers).
* `mean`: Fill with training-set mean.
* `zero`: Assume missing data implies zero wealth (Aggressive).

### Missingness Indicators
By setting `add_indicator=True`, the imputer appends a `<col>__was_missing` column. This is critical because for many vendors, **missing data is a signal** (e.g., the donor might be a patient who hasn't been screened yet or is new to the database).

```python
from philanthropy.preprocessing import WealthScreeningImputer

imputer = WealthScreeningImputer(wealth_cols=["net_worth", "real_estate"], strategy="median", add_indicator=True)
# X_out = imputer.fit_transform(X)
```

## Using `WealthScreeningImputerKNN`

If your missing data can be estimated from similar donors, use KNN imputation:

```python
from philanthropy.preprocessing import WealthScreeningImputerKNN

knn_imputer = WealthScreeningImputerKNN(strategy="knn", n_neighbors=5, add_indicator=True)
# X_out = knn_imputer.fit_transform(X)
```

## Creating Percentiles

Fundraising data is notoriously right-skewed. The `WealthPercentileTransformer` converts raw dollars to 0-100 ranks:

```python
from philanthropy.preprocessing import WealthPercentileTransformer

percentiler = WealthPercentileTransformer(wealth_cols=["net_worth"])
# X_out = percentiler.fit_transform(X)
```
