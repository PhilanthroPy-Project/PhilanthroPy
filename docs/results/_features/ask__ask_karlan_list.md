=== "Karlan and List (real donor file)"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Wealth & demographics | gender, couple and the 2004 vote where the donor lives (no wealth screen) |

    This file has no engagement and mailing history records, so they are not used here.

    These matter most:

    | What raises or lowers the suggested ask | |
    |---|---|
    | largest past gift | ▲ raises the suggested ask |
    | number of past gifts | ▼ lowers the suggested ask |
    | donor is a woman | ▼ lowers the suggested ask |

    Only 3 features had a measurable effect; the rest are in the analyst note below.

    ??? note "For analysts"
        Columns: `prior_gifts`, `highest_previous_amount`, `months_since_last_gift`, `years_since_first_gift`, `female`, `couple`, `red_state`, `red_county`

        | Column | Importance | Direction |
        |---|---|---|
        | `highest_previous_amount` | 20.796 | + |
        | `prior_gifts` | 0.851 | - |
        | `female` | 0.021 | - |
        | `couple` | -0.020 | - |
        | `months_since_last_gift` | -0.056 | + |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `1946e8d50dde3815a976ee1a1a8bed508953d8a5`.

    Results on your own file will differ.
