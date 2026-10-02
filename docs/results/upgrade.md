# $1K upgrade

**Beats the simple rule on our sample data. On the one real file we can test it on today (KDD Cup
1998, at a threshold rescaled to that file's much smaller gifts) it loses. Depends on your data;
try it on your own file before trusting either number.**

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

    Run `score_upgrade_prospects` on a sample donor panel (`make_donor_panel(random_state=42)`,
    one of the five draws averaged above), cutting off at 30 June 2021 and checking fiscal year
    2022. Out of 1,035 mid-level donors held out for validation, the model's top 104 picks (its
    top 10%) included 21 who actually upgraded. Ranking those same 1,035 donors by this year's
    total giving instead would *also* have found 21 at this particular draw, a tie. Picking 104
    of them at random would find about 14.

    **Why this number is not the 22-vs-18 headline above.** The chart above is the average of
    five draws, scored with the benchmark's own feature set built specifically to compare against
    the rule; this worked example runs the actual public `score_upgrade_prospects` function
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

## Which data

Sample (synthetic) donor panel, five random draws averaged, is the only file where this model
wins today. KDD Cup 1998 is the only real file we can test the $1,000 upgrade question on at all
(its gifts are too small to test the real $1,000 threshold, so the threshold above is rescaled;
see the tab). Results on your own file, at your own dollar threshold, will differ from both.

## What the model looks at

--8<-- "results/_features/upgrade__upgrade_synthetic.md"

--8<-- "results/_features/upgrade__upgrade_kdd98.md"
