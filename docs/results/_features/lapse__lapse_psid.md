=== "PSID (household survey)"

    On this file the model looks at 30 things about each household.

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
    | giving this year (this wave on PSID) | ● depends |
    | giving the year before (the wave before on PSID) | ● depends |
    | giving two years before (two waves before on PSID) | ● depends |
    | household wealth, counting home equity | ● depends |
    | consecutive years (or waves) given | ▼ lowers the score |

    ??? note "For analysts"
        Columns: `period_total`, `period_total_prior1`, `period_total_prior2`, `period_trend`, `largest_gift`, `consecutive_periods_given`, `gave_prior1`, `gave_prior2`, `periods_since_first_gift`, `itemized_charitable_contrib_amount`, `family_income`, `wealth1`, `wealth2`, `head_volunteer_hours_annual`, `spouse_volunteer_hours_annual`, `household_volunteer_hours_regular`, `head_volunteer_hours_typical_week`, `spouse_volunteer_hours_typical_week`, `giving_checkpoint_other_2001`, `giving_combo`, `giving_community`, `giving_cultural`, `giving_education`, `giving_environment`, `giving_health`, `giving_international`, `giving_needy`, `giving_other`, `giving_religious`, `giving_youth`

        | Column | Importance | Direction |
        |---|---|---|
        | `period_total` | 0.038 | mixed |
        | `period_total_prior1` | 0.022 | mixed |
        | `period_total_prior2` | 0.014 | mixed |
        | `wealth2` | 0.010 | mixed |
        | `consecutive_periods_given` | 0.008 | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (seed=42, last wave fold). Git SHA: `39d544a09f786786b1e99c0bc67ff7ec0057560a`.

    Results on your own file will differ.
