# Results

Plain-language answers to one question: **does each model actually help you pick
better than the simple rule your shop already uses, or than picking at random?**
No statistics jargon here; everything is in donor counts. For the numbers behind
these pages, see [Model Validation & Benchmarks](../explanation/benchmarks.md).

![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](../assets/results/scoreboard.png#only-light)
![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](../assets/results/scoreboard-dark.png#only-dark)

- **"The rule"** is whatever a fundraising shop already does without a model:
  rank by past giving, ask for what a donor gave last time, mail everyone.
  Every page compares the model against the best version of that rule it
  could find, not a strawman.
- **"Random"** is picking that many donors with no information at all, the
  floor any model or rule should beat.
- **A dot to the right of centre means the model earned its keep** on that
  file; a dot to the left means the simple rule you already run for free
  did better. Losses are shown as often as wins on this page.

Each cell below is that model's verdict on that dataset: **beats the rule**, **about the same as
the rule**, **does not beat the rule**, or, where the file structurally cannot answer the
question, **can't test (reason)**. "not run" means that pairing was never run (either the dataset
doesn't fit the question, or there's nothing new it would show).

A model only counts as beating the rule overall when it wins on two real files. One real win next
to a real loss means **it depends on your data**: that is where leadership upgrade and lapse stand
today, each winning on PSID and losing or tying on the donor files.

| Model | Question it answers | Sample data | KDD Cup 1998 | cup98VAL | DonorsChoose | PSID |
|---|---|---|---|---|---|---|
| [Leadership upgrade ($1,000+)](leadership.md) | Which mid-level donors are about to become $1,000+ donors? | Beats the rule | Loses | not run | About the same as the rule, ahead at the very top | Beats the rule |
| [Response](response.md) | Who is most likely to give again next year? | Does not beat the rule | About the same as the rule | Beats the rule | can't test (no mailing/appeal log) | can't test (no mailing/appeal log) |
| [Lapse](lapse.md) | Which donors are about to stop giving? | Does not beat the rule | About the same as random | not run | Loses, small margin (retention read: about the same, ahead at the top) | Beats the rule (retention read: about the same) |
| [Suggested ask](ask.md) | How much should we ask a donor for? | Does not beat the rule | Does not beat the rule | not run | Does not beat the rule | About the same as the rule |
| [Planned giving](planned_giving.md) | Which donors look like bequest prospects? | Not yet tested on real bequest data | can't test (no bequest-intent label) | not run | can't test (no bequest-intent signal) | can't test (no bequest-intent signal) |
| [Who to mail](who_to_mail.md) | Is it worth mailing this donor at all? | not run | Beats mailing everyone | Beats mailing everyone | can't test (no per-contact mailing cost) | can't test (no per-contact mailing cost) |

## How we tested each one

For every model we picked a cutoff date, gave the model only the gifts recorded
up to that date, and then checked what the same donors actually did in the
following fiscal year. Every model is compared against a simple rule a
fundraising shop already uses without any model (rank by past giving, ask for
what they gave last time, mail everyone) and against picking at random. We ran
this on a synthetic sample donor panel we generate ourselves (five different
random draws, averaged, so one lucky sample can't flatter the numbers), and
three real public files: [KDD Cup
1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html), a 1990s
direct-mail history from a real nonprofit; DonorsChoose Open Data (ICPSR
37898, doi:10.3886/ICPSR37898.v1), a real giving history for an
education-crowdfunding platform (a random 10% sample of individual donors,
walk-forward by fiscal year); and the Panel Study of Income Dynamics (PSID),
a survey of US households every two years in which the household head reports
the household's total giving to all charities (survey waves 2015 to 2021
tested in turn, training on every earlier wave back to 2001). PSID data are
not redistributed here; to rerun those numbers, download your own extract from
the PSID Data Center. Every number on these pages is
produced by `scripts/make_results_pages.py`, committed alongside its output in
`docs/assets/results/`, so anyone can regenerate them.

Panel Study of Income Dynamics, public use dataset. Produced and distributed by
the Survey Research Center, Institute for Social Research, University of
Michigan, Ann Arbor, MI.
