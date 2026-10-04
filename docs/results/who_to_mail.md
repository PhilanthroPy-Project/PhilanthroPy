# Who to mail

**Use the model (tested on one organisation so far).** Both files below come from one charity's 1997 mailing.

**We skipped 9,920 of 28,624 letters and still raised $1,393 more.** On a real
donor file, mailing only the donors the model expects to be worth the postage
beats mailing everyone, in dollars raised and in letters saved.

Featured on a real donor file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html).
We pretended it was 1 June 1997 and asked: for each donor, is a $0.68 mailing
worth sending, or should we skip it?

![Net revenue against how many donors are mailed, most to least likely to respond, KDD Cup 1998](../assets/results/who_to_mail.png#only-light)
![Net revenue against how many donors are mailed, most to least likely to respond, KDD Cup 1998](../assets/results/who_to_mail-dark.png#only-dark)

The chart ranks the 28,624 donors held out for this test from most to least
likely to respond, and tracks net revenue as you mail further down that list.
The model's own stopping point, mailing 18,704 of them, is marked: past that
point, the next donor's expected gift no longer covers the $0.68 mailing
cost, so mailing further loses money. That stopping point brings in $4,542
after costs, against $3,149 for mailing everyone.

**Beats mailing everyone.** Skipping the donors least likely to respond
raised more money net, not less, even though fewer pieces went out. The
curve also answers "what if our mailing costs more (or less) than $0.68?":
a higher cost per piece shifts the stopping point left, without needing a
new chart.

**Checked again on a file the model never saw at all.** The chart above
scores a random 30% slice of the same file the model was fit on. KDD Cup
1998 also released a second, entirely separate file for exactly this
purpose: `cup98VAL`, 96,367 more donors, with its answer key (`valtargt`)
withheld until after the original competition and never touched during
fitting or the 55/15/30 split above. Scored on that file with the same
$0.68 rule: mailing the 62,703 of 96,367 donors the model expected to be
worth it brought in $13,764 after costs, against $10,560 for mailing
everyone. Same shape as the chart above, on a file the model has never
seen in any capacity.

![Net revenue against how many donors are mailed, most to least likely to respond, cup98VAL](../assets/results/who_to_mail_cup98val.png#only-light)
![Net revenue against how many donors are mailed, most to least likely to respond, cup98VAL](../assets/results/who_to_mail_cup98val-dark.png#only-dark)

## Which data

Real file: [KDD Cup 1998](https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html),
a 1990s direct-mail history; gift sizes there are small (a few dollars to a
few hundred), so its dollar figures will not resemble a major-gift program.
Results on your own file, and at your own mailing cost, will differ.

**Not testable on DonorsChoose or PSID.** Neither file records a per-contact mailing cost, so
cost-aware selection has no cost side to weigh.

## What the model looks at

"Mail if expected gift beats the cost" multiplies a response model's score by
a suggested gift amount, so there is no single ranking to attribute to one
feature list. The drivers below are for the response half of that decision
only; see [Suggested ask](ask.md#what-the-model-looks-at) for the amount
half's own drivers.

--8<-- "results/_features/who_to_mail__who_to_mail_kdd98.md"

--8<-- "results/_features/who_to_mail__who_to_mail_cup98val.md"
