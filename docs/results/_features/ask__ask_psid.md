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

    | What raises or lowers the suggested ask | |
    |---|---|
    | this wave's giving | ▲ raises the suggested ask |
    | last wave's giving | ● depends |
    | giving to religious causes | ▲ raises the suggested ask |
    | charitable deduction claimed on taxes | ● depends |
    | largest single cause this wave | ▲ raises the suggested ask |

    ??? note "For analysts"
        Columns: `total_giving`, `itemized_charitable_contrib_amount`, `family_income`, `wealth1`, `wealth2`, `head_volunteer_hours_annual`, `spouse_volunteer_hours_annual`, `household_volunteer_hours_regular`, `head_volunteer_hours_typical_week`, `spouse_volunteer_hours_typical_week`, `giving_checkpoint_other_2001`, `giving_combo`, `giving_community`, `giving_cultural`, `giving_education`, `giving_environment`, `giving_health`, `giving_international`, `giving_needy`, `giving_other`, `giving_religious`, `giving_youth`, `largest_giving_category`, `prior_wave_total`, `trend`, `waves_given_streak`

        | Column | Importance | Direction |
        |---|---|---|
        | `total_giving` | 821.814 | + |
        | `prior_wave_total` | 317.692 | mixed |
        | `giving_religious` | 295.694 | + |
        | `itemized_charitable_contrib_amount` | 66.727 | mixed |
        | `largest_giving_category` | 57.953 | + |

        Method: permutation importance (`neg_mean_absolute_error`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (seed=42, last wave fold). Git SHA: `17483eaac7c0ba11b1da760f9f8d6b6199804e3a`.

    Results on your own file will differ.
