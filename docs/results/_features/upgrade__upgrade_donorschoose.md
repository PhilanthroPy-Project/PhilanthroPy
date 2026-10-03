=== "DonorsChoose (real donor file)"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |

    This file has no engagement, wealth & demographics, and mailing history records, so they are not used here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | this year's giving | ▲ raises the score |
    | last year's giving | ▲ raises the score |
    | months since last gift | ▼ lowers the score |
    | giving trend, this year vs last | ▲ raises the score |
    | largest gift this year | ▲ raises the score |

    ??? note "For analysts"
        Columns: `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `fy_total` | 0.090 | + |
        | `fy_total_prior1` | 0.021 | + |
        | `months_since_last_gift` | 0.013 | - |
        | `fy_trend` | 0.007 | + |
        | `largest_gift` | 0.005 | + |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (subsample=0.1, seed=42, last fiscal-year fold). Git SHA: `169aea3cdb71091fa4bb31f877c89bc3b0342e20`.

    Results on your own file will differ.
