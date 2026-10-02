=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 17 things about each donor; these matter most.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Wealth & demographics | wealth screening and demographic data |
    | Mailing history | how they have responded to past mailings |

    This file has no momentum, engagement records, so they are not used here.

    | What raises or lowers the score | |
    |---|---|
    | last gift amount | ▲ raises the score |
    | average gift amount | ▲ raises the score |
    | number of gifts | ▼ lowers the score |
    | lifetime giving | ▲ raises the score |
    | largest gift ever | ▲ raises the score |

    ??? note "For analysts"
        Columns: `AGE`, `INCOME`, `WEALTH1`, `WEALTH2`, `NUMCHLD`, `RAMNTALL`, `NGIFTALL`, `LASTGIFT`, `AVGGIFT`, `MAXRAMNT`, `MINRAMNT`, `TIMELAG`, `HOMEOWNER`, `rfm_recency`, `rfm_frequency`, `rfm_monetary`, `rfm_tenure`

        | Column | Importance | Direction |
        |---|---|---|
        | `LASTGIFT` | 2.879 | + |
        | `AVGGIFT` | 0.834 | + |
        | `rfm_frequency` | 0.176 | - |
        | `rfm_monetary` | 0.134 | + |
        | `MAXRAMNT` | 0.130 | + |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Configurations compared: 1. Git SHA: `fc5b61047663274deb522e370c66af1aea975beb`.

    Results on your own file will differ.
