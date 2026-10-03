=== "PSID (household survey)"

    On this file the model looks at 26 things about each household.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |
    | Year-over-year change | whether this year's giving is up or down from last year |
    | Engagement | events, volunteering and other non-gift contact |
    | Wealth & demographics | self-reported household income and wealth (survey answers, not a wealth screen) |

    This file has no mailing history records, so it is not used here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | this wave's giving | ● depends |
    | last wave's giving | ● depends |
    | consecutive waves given | ▼ lowers the score |
    | household wealth, counting home equity | ● depends |
    | charitable deduction claimed on taxes | ● depends |

    ??? note "For analysts"
        Columns: `total_giving`, `itemized_charitable_contrib_amount`, `family_income`, `wealth1`, `wealth2`, `head_volunteer_hours_annual`, `spouse_volunteer_hours_annual`, `household_volunteer_hours_regular`, `head_volunteer_hours_typical_week`, `spouse_volunteer_hours_typical_week`, `giving_checkpoint_other_2001`, `giving_combo`, `giving_community`, `giving_cultural`, `giving_education`, `giving_environment`, `giving_health`, `giving_international`, `giving_needy`, `giving_other`, `giving_religious`, `giving_youth`, `largest_giving_category`, `prior_wave_total`, `trend`, `waves_given_streak`

        | Column | Importance | Direction |
        |---|---|---|
        | `total_giving` | 0.042 | mixed |
        | `prior_wave_total` | 0.031 | mixed |
        | `waves_given_streak` | 0.010 | - |
        | `wealth2` | 0.006 | mixed |
        | `itemized_charitable_contrib_amount` | 0.005 | mixed |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (seed=42, last wave fold). Git SHA: `b0122490e6902ce1e5c9f00e3f0d142874a1c72f`.

    Results on your own file will differ.
