# How to read these pages

Every Results page answers one question: on donors the model has never
seen, does it pick better than the simple rule your shop already uses for
free, and better than picking at random?

## The simple rule your shop already uses

What a fundraising shop does without a model. Each question has its own:

| Question | The simple rule |
|---|---|
| Who will move up to $1,000+ next year? | Rank by this year's giving. |
| Who will give again? | Rank by past giving. |
| Who is about to stop giving? | Rank by years since the last gift. |
| How much should we ask for? | Ask for what they gave last time. |
| Is this donor worth mailing? | Mail everyone. |

Each page tries a short list of these rules, fixed in code before any run,
and compares the model against whichever did best on that file. The model is
never compared against a strawman.

## Picking at random

Picking the same number of donors with no information at all. What random
finds is simply the file's base rate: if 16 out of every 100 donors give
again, a random 100 contains about 16 who give again. A model or rule that
does not beat random has learned nothing.

## Top 1%, 5% and 10%

A list length, not a score. Rank every donor, then take the first 1%, 5% or
10% of names: the size of list a gift officer or a mailing can actually work.
On a file of 10,000 donors, the top 10% is the first 1,000 names.

A worked example from DonorsChoose ("who keeps giving next year", on the
[Lapse](lapse.md) page), scaled to a 10,000-donor file:

- **Random:** a random 1,000 names include about 160 who give again.
- **The simple rule:** the rule's first 1,000 names include about 450.
- **The model:** the model's first 1,000 names include about 500.

Both lists are far ahead of random. The model is ahead of the rule by about
50 donors in 1,000, which is the kind of gap the next section is about.

## About the same

The model is retrained on several test years (or several random draws), and
its number moves a little from run to run. When the gap between the model and
the rule is smaller than that run-to-run range, we call it about the same: you
would not see the difference on your own file, so keep the rule, which is free
and easy to explain.

## Bottom lines

Each model page ends its findings with one of five bottom lines:

| Bottom line | When we say it |
|---|---|
| **Use the model** | The model beats the simple rule on real files, across the whole list. |
| **Use the model for your top slice only** | The model is ahead only for the first names on the list (top 1% or 5%), about the same further down. |
| **Use the simple rule** | The rule does as well as the model, or better. |
| **Use the retention list** | Nearly everyone on the file lapses, so the useful list is the reverse one: the donors most likely to keep giving. |
| **Can't tell yet** | One real file says yes and another says no, or no real file can test the question. |

### Words on the index today

The [Results](index.md) table still uses the older per-file words below; they
are being replaced by the five bottom lines above. Each maps to the bottom line
it becomes:

| Word on the index today | What it means | Becomes |
|---|---|---|
| **Beats the rule** | The model's list finds clearly more of the right donors than the best simple rule. | Use the model |
| **About the same as the rule** | The gap is inside the run-to-run range. Where a cell adds "ahead at the very top", the model leads only at the top 1% or 5%. | Use the simple rule, or Use the model for your top slice only |
| **Does not beat the rule** / **Loses** | The rule does as well or better. | Use the simple rule |
| **About the same as random** | Neither the model nor the rule tells donors apart much, usually because almost everyone lapses. | Use the retention list |
| **can't test (reason)** | The file has nothing to check the answer against, for example no mailing log for a response model. | Can't tell yet |
| **not run** | The pairing was never run: the dataset doesn't fit the question, or there is nothing new it would show. | (no bottom line) |

## Why the numbers can be trusted

Each test pretends it is an earlier date: the model sees only the gifts up to
that date, then is scored on what the same donors did afterwards. Every number
is regenerated from `scripts/make_results_pages.py`, losses are published next
to wins, and the same test can be run on your own file. The files themselves
are described in [The datasets behind these pages](datasets.md).
