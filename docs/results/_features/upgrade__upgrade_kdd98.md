=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 8 things about each donor; these matter most.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Momentum | whether their giving has been rising over the last few years |

    This file has no engagement, wealth & demographics, mailing history records, so they are not used here.

    | What raises or lowers the score | |
    |---|---|
    | largest gift this year | ▲ raises the score |
    | this year's giving | ▲ raises the score |
    | months since last gift | ▲ raises the score |
    | last year's giving | ● depends |
    | giving two years ago | ● depends |

    ??? note "For analysts"
        Columns: `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `largest_gift` | 0.101 | + |
        | `fy_total` | 0.080 | + |
        | `months_since_last_gift` | 0.048 | + |
        | `fy_total_prior1` | 0.000 | mixed |
        | `fy_total_prior2` | 0.000 | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward by fiscal year (train FY1994, test FY1995). Configurations compared: 1. Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
