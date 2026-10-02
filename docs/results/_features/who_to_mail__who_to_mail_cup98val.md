=== "cup98VAL (real donor file, never seen by the model)"

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
    | last gift amount | ▼ lowers the score |
    | number of gifts | ▲ raises the score |
    | lifetime number of gifts | ▲ raises the score |
    | household income bracket | ▲ raises the score |
    | donor's age | ● depends |

    ??? note "For analysts"
        Columns: `AGE`, `INCOME`, `WEALTH1`, `WEALTH2`, `NUMCHLD`, `RAMNTALL`, `NGIFTALL`, `LASTGIFT`, `AVGGIFT`, `MAXRAMNT`, `MINRAMNT`, `TIMELAG`, `HOMEOWNER`, `rfm_recency`, `rfm_frequency`, `rfm_monetary`, `rfm_tenure`

        | Column | Importance | Direction |
        |---|---|---|
        | `LASTGIFT` | 0.024 | - |
        | `rfm_frequency` | 0.007 | + |
        | `NGIFTALL` | 0.006 | + |
        | `INCOME` | 0.005 | + |
        | `AGE` | 0.005 | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Configurations compared: 1. Git SHA: `fc5b61047663274deb522e370c66af1aea975beb`.
        Same response model and features as the KDD Cup 1998 tab; cup98VAL supplies new test donors on the same columns, not new columns.

    Results on your own file will differ.
