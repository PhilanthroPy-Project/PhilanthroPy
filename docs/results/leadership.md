# Leadership upgrade ($1,000+)

Which donors giving $100 to $999 this fiscal year will give $1,000 or more next fiscal year?
$1,000 is the default leadership level; set your own with `threshold=` and `band=` in
`score_leadership_prospects` (or `--threshold` / `--band` on the command line).

**Depends on your data.** It beats the simple rule on our sample data, and on the three real
files we can test it on today the results split: it beats the rule on PSID household survey
data, is about the same as the rule on DonorsChoose (ahead only at the very top of the list), and
loses on KDD Cup 1998 (threshold rescaled to that file's much smaller gifts). One real win next
to a real loss is not enough to say it beats the rule in general; try it on your own file before
trusting any of these numbers.

=== "Sample data"

    We pretended it was 30 June 2021: the model only saw gifts up to that date, then we checked
    what mid-level donors actually did in fiscal year 2022 (1 July 2021 to 30 June 2022),
    specifically which of them crossed $1,000 for the first time.

    Of our top 10% of picks, 22 out of every 100 crossed $1,000. Ranking by this year's giving
    total alone found 18 out of every 100. Picking at random finds about 12 out of every 100.

    ![Model vs. best simple rule vs. random, top 1/5/10% of picks](../assets/results/upgrade_topn.png#only-light)
    ![Model vs. best simple rule vs. random, top 1/5/10% of picks](../assets/results/upgrade_topn-dark.png#only-dark)

    **Beats the simple rule here.**

    ### Worked example

    Run `score_leadership_prospects` on a sample donor panel (`make_donor_panel(random_state=42)`,
    one of the five draws averaged above), cutting off at 30 June 2021 and checking fiscal year
    2022. Out of 1,035 mid-level donors held out for validation, the model's top 104 picks (its
    top 10%) included 21 who actually upgraded. Ranking those same 1,035 donors by this year's
    total giving instead would *also* have found 21 at this particular draw, a tie. Picking 104
    of them at random would find about 14.

    **Why this number is not the 22-vs-18 headline above.** The chart above is the average of
    five draws, scored with the benchmark's own feature set built specifically to compare against
    the rule; this worked example runs the actual public `score_leadership_prospects` function
    end to end on one of those same five draws (the first), which has its own feature set and
    model settings and so gets its own answer. Both beat the naive "pick at random" floor; take
    the five-draw average as the more reliable estimate of the gap, and this one seed as proof
    the shipped function itself works, not a second measurement of the same number.

    The chart below averages the same per-decile breakdown across all five draws (D1 = the 10%
    the model liked most, D10 = the 10% it liked least), with the range across draws shown as a
    line through each bar. The top decile clearly stands out; the groups below it decline toward
    the overall rate. Use the model to pick your top slice, not to rank the whole file:

    ![Upgrade rate by decile, 5 draws averaged](../assets/results/upgrade_deciles.png#only-light)
    ![Upgrade rate by decile, 5 draws averaged](../assets/results/upgrade_deciles-dark.png#only-dark)

=== "KDD Cup 1998 (real donor file)"

    KDD98's gifts top out far lower than a major-gift program's, so the threshold is rescaled to
    $50 (roughly this file's 93rd percentile of annual per-donor giving) instead of $1,000, and
    the test is walk-forward by fiscal year rather than the sample data's random split (this file
    has only one real fiscal-year transition to test on; see "Which data" below).

    Of the model's top 10% of picks, 11 out of every 100 crossed $50. The best rule (the largest
    single gift already in the eligible band) found 12 out of every 100. At the very top of the
    list the gap is bigger and goes the other way: the model's top 1% found 17 in 100 against 23
    in 100 for the rule.

    ![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/upgrade_kdd98.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/upgrade_kdd98-dark.png#only-dark)

    **Loses here.** On this file, at this threshold, ranking by the size of a donor's biggest
    eligible gift beats the model, especially for the very top of the list.

    Adding this year's giving trend vs. last year's as its own feature made no difference here at
    all: the model's top 1/5/10% hit rates (17, 14, and 11 out of 100) are identical with and
    without it, down to the last fraction of a percent. On this file the model gives those extra
    columns zero weight, not "no data to add": the columns are there, the fit just never uses
    them.

=== "DonorsChoose (real donor file)"

    [DonorsChoose Open Data](https://www.icpsr.umich.edu/web/ICPSR/studies/37898) (ICPSR 37898,
    doi:10.3886/ICPSR37898.v1) is a real giving history for an education-crowdfunding platform,
    far larger than KDD Cup 1998. We used a random 10% sample of individual ("citizen") donors
    (about 40,000 donor-year rows across the test years) and tested fiscal years 2015 through
    2018 in turn, training on every earlier year each time.

    Of the model's top 10% of picks, 9 out of every 100 crossed $1,000 the following fiscal year.
    The best of the 3 simple rules checked here (this year's total alone, the winner in all 4
    test years) found 8 out of every 100. At the very top of the list the model is clearly ahead:
    its top 1% found 28 in 100 against 20 in 100 for the rule. Upgrading to $1,000+ is rare in
    this file: only about 1 in 100 donors in the test years did it at all.

    ![Model vs. best simple rule, top 1/5/10% of picks, DonorsChoose](../assets/results/upgrade_donorschoose.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, DonorsChoose](../assets/results/upgrade_donorschoose-dark.png#only-dark)

    **About the same as the rule overall**, though clearly ahead at the very top of the list.

    Adding this year's giving trend vs. last year's as its own feature made essentially no
    difference here either: the model's top 10% hit rate is 9 out of 100 with it, 9 without. The
    top 1% shifts from 28 to 25 out of 100, but that group is only about 100 donors per test year,
    so a 3-donor difference is too small to call a real change either way: this is a single fit on
    one file, with no seed range to check it against.

    Never a donor ID or single-donor example here: every number above is an aggregate over the
    sampled donor population.

=== "PSID (household survey)"

    The [Panel Study of Income Dynamics](https://psidonline.isr.umich.edu/) (PSID) is a
    long-running survey of US households, interviewed every two years (one "wave" every two
    years). Each wave asks the household head how much the household gave to all charities over
    the past year, broken down by cause. So these are self-reported household totals across every
    charity, not one organization's donor file. We took households that gave $100 to $999 in a
    wave and checked which of them reported $1,000 or more in the next wave. We tested the 2015,
    2017, 2019 and 2021 waves in turn, training on every earlier wave back to 2001 each time:
    between 1,315 and 1,751 households per test wave, and about 16 in 100 of them crossed $1,000.

    Of the model's top 10% of picks, 41 out of every 100 crossed $1,000 by the next wave. The
    best of the 3 simple rules checked here (this wave's total alone, the winner in all 4 test
    waves) found 32 out of every 100. The model was ahead at the top 10% in every test wave,
    though only barely in 2019 (37 vs 36). At the top 5% it found 45 against 35.

    The top 1% looks even better for the model (48 in 100 against 23 for the rule), but that
    group is only 13 to 18 households per test wave, so treat it as a hint, not a measurement.

    ![Model vs. best simple rule, top 1/5/10% of picks, PSID](../assets/results/upgrade_psid.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, PSID](../assets/results/upgrade_psid-dark.png#only-dark)

    **Beats the rule here**, judged on the top 10% of the list.

    Adding the giving trend between waves as its own feature moved the top 10% from 41 to 42
    out of 100, too small to call a change. The top 1% moved from 48 to 62, but on 13 to 18
    households per wave that is a few households either way, so it is not evidence of anything.

    Only aggregates are shown here: no household IDs, no single-household examples. PSID data are
    not redistributed with this project; to rerun these numbers, download your own extract from
    the PSID Data Center.

    Panel Study of Income Dynamics, public use dataset. Produced and distributed by the Survey
    Research Center, Institute for Social Research, University of Michigan, Ann Arbor, MI.

## Which data

Sample (synthetic) donor panel, five random draws averaged, and PSID are the two files where
this model beats the rule today. KDD Cup 1998, DonorsChoose and PSID are the three real files we
can test the $1,000 upgrade question on (KDD98's gifts are too small to test the real $1,000
threshold, so the threshold on that tab is rescaled; DonorsChoose's and PSID's amounts are large
enough to use $1,000 as-is). Results on your own file, at your own dollar threshold, will differ
from all four.

## What the model looks at

--8<-- "results/_features/upgrade__upgrade_synthetic.md"

--8<-- "results/_features/upgrade__upgrade_kdd98.md"

--8<-- "results/_features/upgrade__upgrade_donorschoose.md"

--8<-- "results/_features/upgrade__upgrade_psid.md"
