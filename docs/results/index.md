# Results

Plain-language answers to one question: **does each model actually help you pick
better than the simple rule your shop already uses, or than picking at random?**
No statistics jargon here; everything is in donor counts. For the numbers behind
these pages, see [Model Validation & Benchmarks](../explanation/benchmarks.md). New
here? Start with [How to read these pages](how_to_read.md); the files used are
described in [The datasets behind these pages](datasets.md).

![How each model compares to the simple rule your shop already uses: one dot per real file, the model's result as a multiple of the rule's, with a line for its range; left of 1 means the rule did better](../assets/results/scoreboard.png#only-light)
![How each model compares to the simple rule your shop already uses: one dot per real file, the model's result as a multiple of the rule's, with a line for its range; left of 1 means the rule did better](../assets/results/scoreboard-dark.png#only-dark)

- **"The rule"** is whatever a fundraising shop already does without a model:
  rank by past giving, ask for what a donor gave last time, mail everyone.
  Every page compares the model against the best version of that rule it
  could find, not a strawman.
- **"Random"** is picking that many donors with no information at all, the
  floor any model or rule should beat.
- **A filled dot right of the 1x line means the model earned its keep** on
  that file; a filled dot left of it means the simple rule you already run
  for free did better, and a hollow dot means the two are too close to call.
  Losses are shown as often as wins on this page.

Each cell below is that model's result on that real file, in donors out of 100 among the
top 10% it picks: **beats the rule**, **about the same as the rule**, or **loses to the rule**,
or, where the file structurally cannot answer the question, **can't test (reason)**. "not run"
means that pairing was never run. A model only beats the rule when the whole range of its
margin over the rule, across resamples or test years, lands on the model's side; if the range
straddles zero the cell says **about the same**. The synthetic sample data is not in this table:
it shows the code runs, not that the model helps.

Where nearly every donor lapses, a list of "who will lapse" has nothing useful to say, so the
lapse cell reports the reverse list, who keeps giving, and how many times better than picking
at random it is.

The bottom line follows one fixed rule, the same for every question; [How to read these
pages](how_to_read.md#bottom-lines) states it in full. **Use the model** needs wins on two
independent sources (two organisations' files, or one file in every one of several test years)
and no loss on a file that counts; wins from one organisation only read **Use the model (tested
on one organisation so far)**. A win next to a loss reads **Can't tell yet**. The KDD Cup
1998 upgrade cell is a $50 proxy for the $1,000 question, so it is shown but not counted. This
table and the scoreboard above are generated from the same results file as every number on
these pages.

--8<-- "results/_verdicts/index_table.md"

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
