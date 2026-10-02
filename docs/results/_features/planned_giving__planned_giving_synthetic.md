=== "Sample data"

    On this file the model looks at 3 things about each donor; these matter most.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no momentum, engagement, wealth & demographics, mailing history records, so they are not used here.

    | What raises or lowers the score | |
    |---|---|
    | lifetime giving | ▲ raises the score |
    | this year's gift | ▼ lowers the score |
    | number of gifts | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 0.182 (range 0.155 to 0.215) | + |
        | `recent` | 0.017 (range 0.001 to 0.050) | - |
        | `n` | 0.001 (range -0.003 to 0.005) | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Configurations compared: 1. Git SHA: `fc5b61047663274deb522e370c66af1aea975beb`.

    Results on your own file will differ.
