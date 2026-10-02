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
    | this year's gift | ● depends |
    | number of gifts | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 25.692 (range 14.151 to 37.338) | + |
        | `recent` | 5.181 (range 1.101 to 9.047) | mixed |
        | `n` | 0.462 (range -2.434 to 3.683) | - |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Configurations compared: 1. Git SHA: `fc5b61047663274deb522e370c66af1aea975beb`.

    Results on your own file will differ.
