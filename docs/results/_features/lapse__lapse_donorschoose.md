=== "DonorsChoose (real donor file)"

    On this file the model looks at 11 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |

    This file has no engagement, wealth & demographics, and mailing history records, so they are not used here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | number of gifts this year | ▼ lowers the score |
    | months since last gift | ▲ raises the score |
    | giving this year (this wave on PSID) | ▼ lowers the score |
    | giving the year before (the wave before on PSID) | ▼ lowers the score |
    | consecutive years (or waves) given | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `period_total`, `period_total_prior1`, `period_total_prior2`, `period_trend`, `largest_gift`, `consecutive_periods_given`, `gave_prior1`, `gave_prior2`, `periods_since_first_gift`, `gift_count`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `gift_count` | 0.022 | - |
        | `months_since_last_gift` | 0.022 | + |
        | `period_total` | 0.021 | - |
        | `period_total_prior1` | 0.002 | - |
        | `consecutive_periods_given` | 0.001 | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (subsample=0.1, seed=42, last fiscal-year fold). Git SHA: `d857e8aac5ff4cdbb8c91450cec7583480c1d7af`.

    Results on your own file will differ.
