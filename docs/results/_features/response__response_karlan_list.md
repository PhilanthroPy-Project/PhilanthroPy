=== "Karlan and List (real donor file)"

    On this file the model looks at 8 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Wealth & demographics | gender, couple and the 2004 vote where the donor lives (no wealth screen) |

    This file has no engagement and mailing history records, so they are not used here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | number of past gifts | ▲ raises the score |
    | months since last gift | ▼ lowers the score |
    | largest past gift | ● depends |
    | years as a donor | ▼ lowers the score |
    | state voted Republican in 2004 | ▲ raises the score |

    ??? note "For analysts"
        Columns: `prior_gifts`, `highest_previous_amount`, `months_since_last_gift`, `years_since_first_gift`, `female`, `couple`, `red_state`, `red_county`

        | Column | Importance | Direction |
        |---|---|---|
        | `prior_gifts` | 0.122 | + |
        | `months_since_last_gift` | 0.082 | - |
        | `highest_previous_amount` | 0.064 | mixed |
        | `years_since_first_gift` | 0.016 | - |
        | `red_state` | 0.001 | + |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: 55/15/30 stratified (train/validation/test). Git SHA: `fbdb12e026b5950383070a26fe5076b52bd36941`.

    Results on your own file will differ.
