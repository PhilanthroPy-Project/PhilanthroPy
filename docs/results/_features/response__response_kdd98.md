=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 4 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no momentum and engagement records, so they are not used here.
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

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
