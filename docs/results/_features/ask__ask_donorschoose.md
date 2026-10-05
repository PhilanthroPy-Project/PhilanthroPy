=== "DonorsChoose (real donor file)"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |

    This file has no engagement, wealth & demographics, and mailing history records, so they are not used here.

    These matter most:

    | What raises or lowers the suggested ask | |
    |---|---|
    | this year's giving | ▲ raises the suggested ask |
    | largest gift this year | ▲ raises the suggested ask |
    | number of gifts this year | ▲ raises the suggested ask |
    | last year's giving | ▲ raises the suggested ask |
    | giving two years ago | ▲ raises the suggested ask |

    ??? note "For analysts"
        Columns: `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `fy_total` | 68.444 | + |
        | `largest_gift` | 14.647 | + |
        | `gift_count` | 13.859 | + |
        | `fy_total_prior1` | 12.701 | + |
        | `fy_total_prior2` | 1.903 | + |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (subsample=0.1, seed=42, last fiscal-year fold). Git SHA: `d857e8aac5ff4cdbb8c91450cec7583480c1d7af`.

    Results on your own file will differ.
