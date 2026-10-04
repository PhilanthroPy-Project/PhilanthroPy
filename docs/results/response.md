# Response (will they give again?)

**Use the model.** It beats the simple rule on two organisations' real donor files. On one
charity's mailing it wins on the held-out cup98VAL half and is ahead for your top picks (top 1% and
5%) on the KDD Cup 1998 half; those two halves count as one organisation. On a second charity's
letter, the Karlan and List experiment, it wins at every list length checked.

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

=== "Karlan and List (real donor file)"

    About 50,000 past donors to one US charity got the same fundraising letter in 2005, and about
    2 in 100 gave. The model saw only what the charity knew before mailing: number of past gifts,
    largest past gift, months since the last gift, years as a donor, and a few facts from the donor
    record. The comparison rule is the best of most recent first, largest past gift first, most
    gifts first, and an RFM cell score built from those three.

    On a held-out 30% of the donors, the model's top 10% found 8.0 in 100 who gave, against 5.5 in
    100 for the best rule (most gifts first). At the top 1% it found 23 in 100 against 13. Picking at
    random finds about 2 in 100.

    ![Model vs. best simple rule, top 1/5/10% of picks, Karlan and List](../assets/results/response_karlan_list.png#only-light)
    ![Model vs. best simple rule, top 1/5/10% of picks, Karlan and List](../assets/results/response_karlan_list-dark.png#only-dark)

    This is a different charity from the KDD Cup 1998 tab, so it is a second, independent check.
    Results on your own file will differ.

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

## Does a matching-grant offer move some donors more than others?

The Karlan and List letter came in versions: two in three donors, picked at random, were told a
matching grant would multiply their gift. Because the offer was random, the file can test an
uplift model (`UpliftTLearner`), which ranks donors by how much the offer raises their chance of
giving, not by how likely they are to give at all.

Across all held-out donors, the offer raised giving by about 0.4 in 100. Among the 30% the uplift
model ranked highest, it raised giving by about 0.6 in 100, against 0.5 for the best of two simple
rules (most recent donors first). The range on that gap runs from the model about 1 in 100 behind
to about 1 in 100 ahead, so this is **about the same as the rule**. Ranking donors by who the
offer moves most does not yet beat ranking by recency on this file.

Aggregates only; no donor-level figures are shown here.

Karlan, D. and List, J. A. (2007), "Does Price Matter in Charitable Giving? Evidence from a
Large-Scale Natural Field Experiment", *American Economic Review* 97(5): 1774-1793. Data from
openICPSR 113224 (doi:10.3886/E113224V1), CC BY 4.0, copyright American Economic Association 2007.

## What the model looks at

--8<-- "results/_features/response__response_kdd98.md"

--8<-- "results/_features/response__response_cup98val.md"

--8<-- "results/_features/response__response_karlan_list.md"

--8<-- "results/_features/response__response_synthetic.md"
