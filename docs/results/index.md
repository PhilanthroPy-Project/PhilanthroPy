# Results

Plain-language answers to one question: **does each model actually help you pick
better than the simple rule your shop already uses, or than picking at random?**
No statistics jargon here; everything is in donor counts. For the numbers behind
these pages, see [Model Validation & Benchmarks](../explanation/benchmarks.md).

| Model | Question it answers | Verdict |
|---|---|---|
| [$1K upgrade](upgrade.md) | Which mid-level donors are about to become $1,000+ donors? | Beats the simple rule |
| [Response / major gift](response.md) | Who is most likely to give again next year? | Does not beat the simple rule; use the rule instead |
| [Lapse](lapse.md) | Which donors are about to stop giving? | About the same as the simple rule on our sample data; about the same as random on a real donor file |
| [Suggested ask](ask.md) | How much should we ask a donor for? | Beats "ask what they gave last time" on our sample data; loses to it on a real donor file |
| [Planned giving](planned_giving.md) | Which donors look like bequest prospects? | Beats random picking (no simple rule exists yet to compare against) |
| [Who to mail](who_to_mail.md) | Is it worth mailing this donor at all? | Beats mailing everyone |

## How we tested each one

For every model we picked a cutoff date, gave the model only the gifts recorded
up to that date, and then checked what the same donors actually did in the
following fiscal year. Every model is compared against a simple rule a
fundraising shop already uses without any model (rank by past giving, ask for
what they gave last time, mail everyone) and against picking at random. We ran
this on two kinds of data: a synthetic sample donor panel we generate
ourselves (five different random draws, averaged, so one lucky sample can't
flatter the numbers), and a real public file, [KDD Cup
1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html), a 1990s
direct-mail history from a real nonprofit. Every number on these pages is
produced by `scripts/make_results_pages.py`, committed alongside its output in
`docs/assets/results/`, so anyone can regenerate them.
