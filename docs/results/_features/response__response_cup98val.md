=== "cup98VAL (real donor file, never seen by the model)"

    On this file the model looks at 4 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no engagement records, so it is not used here.
    Wealth & demographics and mailing history are in this file, but this model is not given them here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | number of gifts | ▲ raises the score |
    | months since last gift | ▼ lowers the score |
    | lifetime giving | ▼ lowers the score |
    | years as a donor | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `recency`, `frequency`, `monetary`, `tenure`

        | Column | Importance | Direction |
        |---|---|---|
        | `frequency` | 0.073 | + |
        | `recency` | 0.009 | - |
        | `monetary` | 0.008 | - |
        | `tenure` | 0.005 | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `1f042056e7ac724e7d54aa96958c703717c5ec50`.
        Same model and features as the KDD Cup 1998 tab; cup98VAL supplies new test donors on the same columns, not new columns.

    Results on your own file will differ.
