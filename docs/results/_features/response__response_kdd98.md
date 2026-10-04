=== "KDD Cup 1998 (real donor file)"

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

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `39d544a09f786786b1e99c0bc67ff7ec0057560a`.

    Results on your own file will differ.
