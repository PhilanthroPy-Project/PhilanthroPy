=== "Sample data"

    On this file the model looks at 9 things about each donor; these matter most.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Momentum | whether their giving has been rising over the last few years |

    This file has no engagement, wealth & demographics, mailing history records, so they are not used here.

    | What raises or lowers the score | |
    |---|---|
    | last year's giving | ▲ raises the score |
    | giving two years ago | ▲ raises the score |
    | this year's giving | ▲ raises the score |
    | consecutive years given | ▲ raises the score |
    | giving trend, this year vs last | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `fiscal_year`, `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `fy_total_prior1` | 0.051 (range 0.028 to 0.079) | + |
        | `fy_total_prior2` | 0.033 (range 0.017 to 0.049) | + |
        | `fy_total` | 0.024 (range 0.004 to 0.048) | + |
        | `consecutive_years_given` | 0.019 (range 0.004 to 0.034) | + |
        | `fy_trend` | 0.001 (range -0.008 to 0.008) | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (FiscalYearGroupedSplitter, last split). Configurations compared: 1. Git SHA: `fc5b61047663274deb522e370c66af1aea975beb`.

    Results on your own file will differ.
