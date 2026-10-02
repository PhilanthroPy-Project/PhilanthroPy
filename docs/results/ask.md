# Suggested ask

Featured on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
We pretended it was 1 June 1997 and asked the model to suggest an amount for
each donor who went on to give.

Of every 100 suggested amounts, 65 landed within 25% of what the donor
actually gave. The best of the simple rules we compare against here, the
higher of what they gave last time or their average gift, landed within 25%
for 67 out of every 100.

![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98.png#only-light)
![How close is the suggested ask: model vs. best simple rule](../assets/results/ask_kdd98-dark.png#only-dark)

**Does not beat the simple rule; use the rule instead.** On this file, the
simple rule is a better guess than the model's suggestion.

## Which data

Real file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
On our sample (synthetic) donor panel the model also does not beat the best
simple rule (about 23 in 100 within 25%, versus about 24 in 100 for the best
of last gift, max(last gift, average gift), and the median training gift);
use the simple rule on both kinds of data. Results on your own data will
differ from both.

## What the model looks at

--8<-- "results/_features/ask__ask_kdd98.md"

--8<-- "results/_features/ask__ask_synthetic.md"
