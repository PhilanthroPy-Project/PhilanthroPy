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

    | What raises or lowers the score | |
    |---|---|
    | last gift amount | ▼ lowers the score |
    | lifetime number of gifts | ▲ raises the score |
    | number of gifts | ▲ raises the score |
    | donor's age | ● depends |
    | months since last gift | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `AGE`, `INCOME`, `WEALTH1`, `WEALTH2`, `NUMCHLD`, `RAMNTALL`, `NGIFTALL`, `LASTGIFT`, `AVGGIFT`, `MAXRAMNT`, `MINRAMNT`, `TIMELAG`, `HOMEOWNER`, `rfm_recency`, `rfm_frequency`, `rfm_monetary`, `rfm_tenure`

        | Column | Importance | Direction |
        |---|---|---|
        | `LASTGIFT` | 0.026 | - |
        | `NGIFTALL` | 0.008 | + |
        | `rfm_frequency` | 0.007 | + |
        | `AGE` | 0.006 | mixed |
        | `rfm_recency` | 0.005 | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `de2889b263d07df02d35ed429310103c37cc8de6`.

    Results on your own file will differ.
