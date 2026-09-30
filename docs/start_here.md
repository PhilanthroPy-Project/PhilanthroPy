# Start here

**For fundraisers and gift officers, not developers.** If you want to know
whether a model is worth using before you touch any code, read this page
first.

## The problem

Your shop already has a rule for who to call, who to mail, and what to ask
for. Vendors sell scores that claim to beat it. Nobody shows you whether
they actually do, on donors the score has never seen.

## What we did

We tested every model in this library against the rule your shop already
uses, and against picking donors at random, on real donor files. We
pretended it was an earlier date, so the model could only see what a
fundraiser would have known at the time, not what happened afterward.

## What we found, wins and losses together

Two things beat the rule on real files: picking who to mail (more money,
fewer letters) and picking the top slice of likely givers again next year.
Two do not: predicting lapse, and suggesting an ask amount; use the rule you
already have for those, and we say so on those pages. The rest are not
testable on public data yet, and we say that too, rather than show a number
that isn't real.

![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](assets/results/scoreboard.png#only-light)
![How each model compares to the simple rule your shop already uses: one dot per dataset, left of center is worse than the rule, right is better](assets/results/scoreboard-dark.png#only-dark)

See the [full Results pages](results/index.md) for the number behind every
dot on this chart.

## Why you can trust it

Every number on this site is regenerated from code in this repository, not
typed by hand. Losses are published right next to wins, not hidden. The same
test that produced these numbers runs on your own file, so you don't have to
take our word for it.

One thing we found along the way, because it matters: built the usual way,
from a donor's whole giving history, one of these models looked far better
than it actually was. Building its features only from what was known at the
time the prediction would have been made removed that. Every model here is
built the second way.

## What to do Monday

- **Mail smarter, not less.** Use [Who to mail](results/who_to_mail.md) to
  skip the donors least likely to respond, before your next appeal goes out.
- **Prioritize your call list.** Use [Response](results/response.md) to rank
  who to call first for renewed giving.
- **Keep using your own rule for ask amounts and lapse.** [Suggested
  ask](results/ask.md) and [Lapse](results/lapse.md) don't beat what you're
  already doing there, so don't switch.

## Try it on your file

Export your donor data from your CRM, map the columns, and run the same
report on your own donors. See [Get your CRM export
in](how-to/get_your_crm_export_in.md) for the exact column mapping and a
runnable command for your system (Raiser's Edge, Salesforce NPSP,
Bloomerang, DonorPerfect, or CiviCRM).
