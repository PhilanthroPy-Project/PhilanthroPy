=== "Sample data"

    On this file the model looks at 3 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no engagement, wealth & demographics, and mailing history records, so they are not used here.
    Momentum is in this file, but this model is not given it here.

    These matter most:

    | What raises or lowers the suggested ask | |
    |---|---|
    | lifetime giving | ▲ raises the suggested ask |
    | this year's gift | ● depends |

    Only 2 features had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 25.692 (range 14.151 to 37.338) | + |
        | `recent` | 5.181 (range 1.101 to 9.047) | mixed |
        | `n` | 0.462 (range -2.434 to 3.683) | - |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
