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
    | this year's gift | ▼ lowers the score |

    Only 2 features had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 0.182 (range 0.155 to 0.215) | + |
        | `recent` | 0.017 (range 0.001 to 0.050) | - |
        | `n` | 0.001 (range -0.003 to 0.005) | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
