# The datasets behind these pages

Each card has the same five lines. Read the "can and cannot test" line before
carrying a verdict over to your own program: none of these files is your donor
file. Only aggregate numbers from them appear in the docs, never a donor or
household ID or a single-record example.

## KDD Cup 1998

*Real charitable file.*

- **What it is:** a direct-mail fundraising history from one US nonprofit:
  every past mailing and gift per donor, plus whether each donor answered one
  later mailing.
- **Who gave to whom and when:** individuals to one charity, by mail, in the
  mid-1990s.
- **How big:** 95,412 donors.
- **Can and cannot test:** can test response to a mailing, lapse, ask amount
  and whether a donor is worth the cost of a mailing. Cannot test the $1,000
  upgrade question (gifts are small; the page uses a $50 stand-in) or anything
  needing event data. About 95 in 100 donors gave nothing to the held-out
  mailing, so the plain lapse question is lopsided here.
- **Terms:** UCI KDD Archive, free to download; no formal licence is stated.
  Cite it as "KDD Cup 1998"; the original documentation asks that teaching
  material not name the sponsoring organisation.

## cup98VAL

*Real charitable file.*

- **What it is:** the held-out companion to KDD Cup 1998, with the answers
  released separately. No donor in it was used for fitting.
- **Who gave to whom and when:** the same charity and the same mailing as
  KDD Cup 1998, different donors.
- **How big:** 96,367 donors.
- **Can and cannot test:** can test response and who to mail on donors the
  model never saw. Cannot test anything independent of that charity: it is new
  donors, not a new organisation.
- **Terms:** as KDD Cup 1998.

## DonorsChoose Open Data (ICPSR 37898)

*Real charitable file.*

- **What it is:** a real giving history from an education crowdfunding
  platform, read from a file you download yourself.
- **Who gave to whom and when:** individuals to classroom projects, online;
  the pages test fiscal years 2015 to 2018, one year at a time.
- **How big:** the pages use a random 10% sample of individual donors, about
  35,000 to 52,000 donors per test year for lapse.
- **Can and cannot test:** can test the $1,000 upgrade, lapse and retention,
  and ask amount, each in four separate test years. Cannot test response (no
  mailing or appeal log), who to mail (no mailing cost) or planned giving (no
  bequest signal). Dates are recorded by month, and most donors give once.
- **Terms:** research and statistical analysis only, no redistribution, no
  study of individual subjects (doi:10.3886/ICPSR37898.v1). The library reads
  it from a local path and never downloads or ships it.

## PSID Philanthropy Panel Study

*Real household survey.*

- **What it is:** household giving and volunteering in the Panel Study of
  Income Dynamics, self-reported by the household head every two years. Read
  from an extract you build and download yourself.
- **Who gave to whom and when:** US households to all charitable causes, not
  one organisation's donors; waves from 2001, test waves 2015, 2017, 2019 and
  2021, each trained on every earlier wave.
- **How big:** about 3,100 to 3,800 giving households per test wave for lapse.
- **Can and cannot test:** can test the $1,000 upgrade, lapse and ask amount
  at the household level. Cannot test response, who to mail or planned giving
  (no appeal log, no mailing cost, no bequest signal). Answers are two years
  apart and from memory, not a gift ledger.
- **Terms:** free after registration with the PSID; no redistribution, no
  attempt to identify anyone, credit to the PSID in any publication. The
  library reads it from a local path and never downloads or ships it.

## Karlan and List matching-grant experiment

*Real charitable file.*

- **What it is:** a randomised fundraising-letter experiment: prior donors to
  one US nonprofit were sent letters with different matching-grant offers and
  ask amounts, and their response was recorded.
- **Who gave to whom and when:** individuals to one charity, by mail, one
  letter in 2005.
- **How big:** 50,083 prior donors, about 2 in 100 of whom gave to the letter.
- **Can and cannot test:** can test response, the amount given and, because
  the matching-grant offer was random, uplift. It is a second organisation for
  response, independent of KDD Cup 1998. Cannot test lapse or upgrade (one
  letter, no repeat years), or who to mail (no mailing cost). The file has a
  largest past gift but no last or average gift.
- **Terms:** openICPSR 113224 (doi:10.3886/E113224V1); data CC BY 4.0, code
  BSD-3, copyright American Economic Association 2007. Cite Karlan and List
  (2007), *American Economic Review* 97(5): 1774-1793. The library reads it
  from a local path (`load_karlan_list`) and never downloads or ships it.

## Sample data

*Sample data: code check only.*

- **What it is:** a donor panel this library generates
  (`philanthropy.datasets.make_donor_panel`), five random draws.
- **Who gave to whom and when:** made-up donors to a made-up organisation.
- **How big:** set by the caller.
- **Can and cannot test:** can show that every model runs end to end. Cannot
  show whether a model helps on real donors; it never earns a verdict.
- **Terms:** none; generated on the fly.
