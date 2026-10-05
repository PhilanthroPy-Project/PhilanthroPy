=== "Sample data"

    On this file the model looks at 3 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no engagement and mailing history records, so they are not used here.
    Wealth & demographics is in this file, but this model is not given it here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | lifetime giving | ▲ raises the score |

    Only 1 feature had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 0.190 (range 0.155 to 0.222) | + |
        | `recent` | 0.019 (range -0.000 to 0.047) | mixed |
        | `n` | -0.003 (range -0.007 to 0.001) | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Git SHA: `d857e8aac5ff4cdbb8c91450cec7583480c1d7af`.

    Results on your own file will differ.
