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
finds is simply the file's base rate: if 36 out of every 100 donors give
again, a random 100 contains about 36 who give again. A model or rule that
does not beat random has learned nothing.

## Top 1%, 5% and 10%

A list length, not a score. Rank every donor, then take the first 1%, 5% or
10% of names: the size of list a gift officer or a mailing can actually work.
On a file of 10,000 donors, the top 10% is the first 1,000 names.

A worked example from DonorsChoose ("who keeps giving next year", on the
[Lapse](lapse.md) page), scaled to a 10,000-donor file:

- **Random:** a random 1,000 names include about 360 who give again.
- **The simple rule:** the rule's first 1,000 names include about 680.
- **The model:** the model's first 1,000 names include about 750.

Both lists are far ahead of random. The model is ahead of the rule by about
70 donors in 1,000, which is the kind of gap the next section is about.

## About the same

The model and the rule are scored on the same donors, and we look at the gap
between them: across many random redraws of the test donors, or across every
test year and random draw. If that gap stays on the model's side every time,
the model beats the rule. If it stays on the rule's side, the model loses. If
it sometimes favours one and sometimes the other, we call it about the same:
you would not reliably see the difference on your own file, so keep the rule,
which is free and easy to explain.

## Bottom lines

Each model page ends its findings with one of six bottom lines. They come from
the index cells by one fixed rule, the same for every question. Only real
files count. Sample data is a code check, and a cell marked "proxy question"
(KDD Cup 1998's $50 version of the $1,000 upgrade question) is shown but not
counted. Where nearly everyone on a file lapses, its "who keeps giving" result
counts instead of its lapse result. KDD Cup 1998 and cup98VAL are two halves of
one charity's mailing, so they count as one source.

| Bottom line | When we say it |
|---|---|
| **Use the model** | The model beats the rule on two independent sources (two organisations' files, or one file in every one of several test years), and loses on none. |
| **Use the model (tested on one organisation so far)** | The model wins and loses nowhere, but every win comes from one organisation's files, each tested once. A second organisation's win would make it "Use the model". |
| **Use the model for your top slice only** | No win and no loss overall, but the model is ahead for the first names on the list (top 1% or 5%). |
| **Use the simple rule** | The rule does as well as the model, or better, on every file that counts. |
| **Use the retention list** | As "Use the model", where every file that counts is one where nearly everyone lapses, so the useful list is the reverse one: the donors most likely to keep giving. |
| **Can't tell yet** | One real file says the model wins and another says it loses, or no real file can test the question. |

### Words in the index cells

Each cell of the [Results](index.md) table gives that model's result on one
real file; the bottom line for the row comes from those cells by a fixed rule.

| Word in a cell | What it means |
|---|---|
| **Beats the rule** | The gap stays on the model's side in every redraw or test year. |
| **About the same as the rule** | The gap goes both ways. Where a cell adds "ahead in the top 1% or 5%", the model's lead holds only for the first names on the list. |
| **Loses to the rule** | The gap stays on the rule's side in every redraw or test year. |
| **Nearly everyone lapses here** | More than 80 in 100 donors lapse, so the cell reports the reverse list, who keeps giving, and how many times better than random it is. |
| **can't test (reason)** | The file has nothing to check the answer against, for example no mailing log for a response model. |
| **not run** | The pairing was never run: the dataset doesn't fit the question, or there is nothing new it would show. |

## Why the numbers can be trusted

Each test pretends it is an earlier date: the model sees only the gifts up to
that date, then is scored on what the same donors did afterwards. Every number
is regenerated from `scripts/make_results_pages.py`, losses are published next
to wins, and the same test can be run on your own file. The files themselves
are described in [The datasets behind these pages](datasets.md).
