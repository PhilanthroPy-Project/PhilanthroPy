=== "KDD Cup 1998 (real donor file)"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |

    This file has no engagement records, so it is not used here.
    Wealth & demographics and mailing history are in this file, but this model is not given them here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | largest gift this year | ▲ raises the score |
    | this year's giving | ▲ raises the score |
    | months since last gift | ▲ raises the score |

    Only 3 features had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `largest_gift` | 0.101 | + |
        | `fy_total` | 0.080 | + |
        | `months_since_last_gift` | 0.048 | + |
        | `fy_total_prior1` | 0.000 | mixed |
        | `fy_total_prior2` | 0.000 | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward by fiscal year (train FY1994, test FY1995). Git SHA: `39d544a09f786786b1e99c0bc67ff7ec0057560a`.

    Results on your own file will differ.
