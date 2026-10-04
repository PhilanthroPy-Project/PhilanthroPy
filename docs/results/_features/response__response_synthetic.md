=== "Sample data (checks the code runs, not that the model works)"

    On this file the model looks at 3 things about each donor.

    | What it knows | What that means |
    |---|---|
    | Giving history | how much and how often they have given in total |
    | Recency | how recently they gave |

    This file has no engagement and mailing history records, so they are not used here.
    Wealth & demographics is in this file, but this model is not given it here.

    These matter most:

    | What raises or lowers the score | |
    |---|---|
    | lifetime giving | ▲ raises the score |
    | this year's gift | ▲ raises the score |

    Only 2 features had a measurable effect; the rest are in the analyst note below.

    ### What adding information did

    | What the model saw | Top 1% | Top 5% | Top 10% |
    |---|---|---|---|
    | only what the model uses today | 82 of 100 | 80 of 100 | 77 of 100 |
    | plus giving streak, time since last gift, last year's gift, biggest gift and tenure | 80 of 100 (between 70 and 87) | 81 of 100 (between 77 and 85) | 77 of 100 (between 75 and 81) |
    | the best simple rule (for comparison) | 91 of 100 | 84 of 100 | 79 of 100 |

    The extra signals made about the same difference at the top of the list.

    ??? note "For analysts"
        Columns: `total`, `n`, `recent`

        | Column | Importance | Direction |
        |---|---|---|
        | `total` | 0.173 (range 0.113 to 0.205) | + |
        | `recent` | 0.020 (range 0.008 to 0.030) | + |
        | `n` | 0.002 (range -0.002 to 0.007) | - |

        Method: permutation importance (`roc_auc`), partial-dependence sign for direction ("mixed" if it changes sign). Split: walk-forward (train on fiscal years < T, test on T). Git SHA: `17483eaac7c0ba11b1da760f9f8d6b6199804e3a`.

    Results on your own file will differ.
