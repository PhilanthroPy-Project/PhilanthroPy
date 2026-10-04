# Response (will they give again?)

**Use the model (tested on one organisation so far).** On one charity's real donor file it beats the simple rule on
the held-out cup98VAL half, and on the KDD Cup 1998 half it is ahead for your top picks (top 1%
and 5%). Both halves come from the same charity's mailing, so this is one organisation, not two.

=== "KDD Cup 1998 (real donor file)"

    We pretended it was 1 June 1997 and checked who gave to that program's next mailing. Only
    about 5 out of every 100 donors did, so random picks find about 5 in 100. The comparison
    rule is the best of lifetime giving, an RFM cell score, and the file's own RFA_2 segment
    (how often the donor gave recently, then their last gift).

    On a held-out 30% of the file, the model's top 1% found 10 in 100 and its top 5% found 10 in
    100, against 7 and 8 in 100 for the best rule. At the top 10% it was about the same as the
    rule (9 in 100 against 8).

    ![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/response_kdd98.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, KDD Cup 1998](../assets/results/response_kdd98-dark.png#only-dark)

    **Checked again on a file the model never saw at all.** `cup98VAL`, a second, entirely
    separate file this program released (96,367 more donors, its answer key withheld until
    after the original competition and never touched during fitting), gives the same shape: the
    model's top 1% found 12 in 100 and its top 10% found 9 in 100, against 9 and 7 in 100 for the
    best rule.

    ![Model vs. best simple rule, top 1/5/10% of picks, cup98VAL](../assets/results/response_cup98val.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, cup98VAL](../assets/results/response_cup98val-dark.png#only-dark)

    So on this real file the model beats the rule for your top picks, on two separate donor
    files from the same program. Results on your own file will differ.

=== "Sample data (checks the code runs, not that the model works)"

    We pretended it was 30 June 2022: the model only saw gifts up to that date, then we checked
    which donors actually gave again in fiscal year 2023 (1 July 2022 to 30 June 2023).

    Of our top 5% of picks, 80 out of every 100 gave again. The best simple rule we compare
    against here (ranking by lifetime giving, or an RFM cell score combining recency, frequency
    and monetary value, whichever does better on each draw) found 84 out of every 100. Picking at
    random finds about 44 out of every 100.

    ![Model vs. best simple rule vs. random, top 1/5/10% of picks](../assets/results/response.png#only-light)
    ![Model vs. best simple rule vs. random, top 1/5/10% of picks](../assets/results/response-dark.png#only-dark)

    **Code check only: sample data is not counted in the verdict.**
    Ranking donors by their own giving history does at least as well as the model at every pick
    size checked. Our sample data is built so that giving again follows past giving almost exactly,
    which is why no model can beat ranking by past giving on it: this tab proves the pipeline runs
    end to end, it is not a test the model can pass. See the real-file tab above for a test that
    actually distinguishes the model from the rule.

Results on your own file will differ from both of these.

**Not testable on DonorsChoose or PSID.** DonorsChoose Open Data has no mailing/appeal log, so a
response model has nothing to predict response to; the PSID giving/volunteering extract has the
same gap.

## What the model looks at

--8<-- "results/_features/response__response_kdd98.md"

--8<-- "results/_features/response__response_cup98val.md"

--8<-- "results/_features/response__response_synthetic.md"
