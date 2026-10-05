=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 17 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Wealth & demographics | wealth screening and demographic data |
    | Mailing history | how they have responded to past mailings |

    This file has no engagement records, so it is not used here.

    These matter most:

    | What raises or lowers the suggested ask | |
    |---|---|
    | last gift amount | ▲ raises the suggested ask |
    | average gift amount | ▲ raises the suggested ask |
    | number of gifts | ▼ lowers the suggested ask |
    | lifetime giving | ▲ raises the suggested ask |
    | largest gift ever | ▲ raises the suggested ask |

    ??? note "For analysts"
        Columns: `AGE`, `INCOME`, `WEALTH1`, `WEALTH2`, `NUMCHLD`, `RAMNTALL`, `NGIFTALL`, `LASTGIFT`, `AVGGIFT`, `MAXRAMNT`, `MINRAMNT`, `TIMELAG`, `HOMEOWNER`, `rfm_recency`, `rfm_frequency`, `rfm_monetary`, `rfm_tenure`

        | Column | Importance | Direction |
        |---|---|---|
        | `LASTGIFT` | 2.733 | + |
        | `AVGGIFT` | 0.795 | + |
        | `rfm_frequency` | 0.229 | - |
        | `rfm_monetary` | 0.213 | + |
        | `MAXRAMNT` | 0.133 | + |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `d857e8aac5ff4cdbb8c91450cec7583480c1d7af`.

    Results on your own file will differ.
