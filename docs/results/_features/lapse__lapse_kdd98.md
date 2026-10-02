=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 3 things about each donor; these matter most.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no momentum, engagement, wealth & demographics, mailing history records, so they are not used here.

    | What raises or lowers the score | |
    |---|---|
    | number of gifts | ● depends |
    | lifetime giving | ▲ raises the score |
    | this year's gift | ● depends |

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `n` | 0.072 | mixed |
        | `total` | 0.022 | + |
        | `recent` | 0.003 | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward across KDD98 promotion periods (train < period N-1, val=N-1, test=N). Configurations compared: 1. Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
