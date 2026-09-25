# CLI Reference

Installing the package puts a `philanthropy` executable on your PATH. See
[Use the CLI](../how-to/use_the_cli.md) for worked invocations.

## `validate` output

Alongside the threshold-0.5 precision/recall/F1, `validate` prints ROC-AUC,
average precision, the base rate, a 10-row decile table, and a top-N
hit-rate/capture line (`--top-n`, a count or a percentage, default 10% of
rows):

```
precision (at threshold 0.5) 1.000
recall    (at threshold 0.5) 1.000
f1        (at threshold 0.5) 1.000
roc_auc                      1.000
average_precision            1.000
base_rate                    0.124

decile  n     positives  hit_rate  lift
     1  146   146        1.000     8.05x
     2  146   35         0.240     1.93x
     3  146   0          0.000     0.00x
     ...
    10  145   0          0.000     0.00x

top 146 of 1457: hit_rate 1.000, captures 0.807 of all positives
```

::: philanthropy.cli
