=== "DonorsChoose (real donor file)"

    On this file the model looks at 16 things about each donor.

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
    | years (or waves) with any giving | ▼ lowers the score |
    | giving the year before (the wave before on PSID) | ▼ lowers the score |
    | giving this year (this wave on PSID) | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `period_total`, `period_total_prior1`, `period_total_prior2`, `period_trend`, `largest_gift`, `consecutive_periods_given`, `gave_prior1`, `gave_prior2`, `periods_since_first_gift`, `gift_count`, `periods_given`, `share_periods_given`, `period_total_slope`, `gift_count_prior1`, `gift_count_trend`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `gift_count` | 0.040 | - |
        | `months_since_last_gift` | 0.020 | + |
        | `periods_given` | 0.019 | - |
        | `period_total_prior1` | 0.017 | - |
        | `period_total` | 0.008 | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (subsample=0.1, seed=42, last fiscal-year fold). Git SHA: `9a237d8447d2bc12d9906204807a9611eb48f06e`.

    Results on your own file will differ.
