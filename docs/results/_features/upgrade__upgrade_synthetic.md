=== "Sample data"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |

    This file has no engagement and mailing history records, so they are not used here.
    Wealth & demographics is in this file, but this model is not given it here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | last year's giving | ▲ raises the score |
    | consecutive years given | ▲ raises the score |

    Only 2 features had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `fy_total`, `fy_total_prior1`, `fy_total_prior2`, `fy_trend`, `largest_gift`, `gift_count`, `consecutive_years_given`, `months_since_last_gift`

        | Column | Importance | Direction |
        |---|---|---|
        | `fy_total_prior1` | 0.047 (range 0.023 to 0.075) | + |
        | `fy_total` | 0.028 (range -0.001 to 0.049) | + |
        | `fy_total_prior2` | 0.018 (range -0.000 to 0.034) | mixed |
        | `consecutive_years_given` | 0.015 (range 0.003 to 0.031) | + |
        | `fy_trend` | 0.000 (range -0.007 to 0.011) | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (FiscalYearGroupedSplitter, last split). Git SHA: `69d1bad4d885b21936de2b47872bdf05ca765970`.

    Results on your own file will differ.
