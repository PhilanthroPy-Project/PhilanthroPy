# Changelog

All notable changes to PhilanthroPy are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [Unreleased]

### Added
- Test coverage now constructs `RFMTransformer(include_tenure=True)`, asserting
  that the optional `tenure` column is emitted, remains at least as large as
  `recency`, and matches `get_feature_names_out`; the default path is pinned to
  keep `tenure` absent unless requested.
- `docs/tutorials/avoiding_temporal_data_leakage.md` now links to the
  real-data replication and states the measured feature-timing leakage cost on
  KDD Cup 1998: walk-forward ROC-AUC 0.482 as of each decision point versus
  0.858 over the whole export, an inflation of +0.376.
- `threshold="pNN"` in `build_leadership_snapshots` and
  `score_leadership_prospects`: the leadership level as the NNth percentile
  of positive donor fiscal-year totals in the training (snapshot) years,
  never their label years, so a small-dollar file and a large-dollar file
  ask the same question. The dollar default (1000) is unchanged; the
  resolved value is in `snapshots.attrs["threshold"]` and
  `report["threshold"]`. The KDD Cup 1998 upgrade tab now uses `"p92"`,
  which resolves to the same $50 it used before, so its numbers do not
  move; the page's "roughly the 93rd percentile" becomes the exact 92nd.
- `score_leadership_prospects` report: `recommended_list_fraction` and
  `recommended_list_size`, the longest top slice of the held-out fold whose
  upgrade rate is still at least 1.5 times the fold's base rate, as a share
  and as a count of today's scored donors.
- Results, leadership: a cross-file transfer check in a collapsed analyst
  block. Fit on DonorsChoose with only scale-free and count columns, the
  model beats PSID's own best rule at the top 10% (36 vs 32 of 100, ahead
  in all 4 test waves) but loses clearly on KDD Cup 1998's $50 proxy (5 vs
  12).
- Results: a "How sure are we?" page for `GiftIntervalCalibrator`. Around
  each real file's ask model, ranges requested at 80%, 90% and 95% are
  calibrated on held-out donors and scored on later ones; on KDD Cup 1998,
  DonorsChoose, PSID and Karlan and List they hold the actual gift within
  3 points of the level asked for. The page also reports how wide the
  ranges are.
- Results: every model page now follows one template, enforced by
  `tests/test_results_page_template.py`: the question, the bold bottom line
  (the only bold on the page, and the one the index computes), results by
  file as tabs with sample data last, "What this means for your file" with
  counts for a 10,000-donor file, a runnable "Try it on your own donors"
  example, then collapsed "How we tested" and "Numbers for analysts". The
  planned giving page's bottom line now reads "Can't tell yet", matching the
  index. The KDD Cup 1998 leadership tab names the two snapshot years its
  $50 threshold is read from.
- `philanthropy.ingest.build_snapshots(gifts, kind=...)`: one labelled
  donor x fiscal-year table for every question, `kind="upgrade"`, `"lapse"`,
  `"response_next_year"` or `"next_amount"`, on one shared column set
  (`period_total` and the two periods before it, `period_trend`,
  `largest_gift`, `consecutive_periods_given`, `gave_prior1`, `gave_prior2`,
  `periods_since_first_gift`, plus `gift_count` and `months_since_last_gift`
  on a gift log). `min_years_given` restricts any kind to donors with that
  many giving years (default 1, the current population). Its core works on a
  donor x period table, where "prior" means the previous period observed in
  the file, so a biennial survey gets the same columns as a gift log. An
  opt-in `scale_free=True` adds ratio and rank columns that do not depend on
  a file's dollar scale. They are not adopted into
  `score_leadership_prospects`: on the E.11a validation folds (upgrade,
  five seeds) they lifted PSID wave 2013 top-10% from 38.9 to 41.6 of 100
  but lowered DonorsChoose FY2014 top-10% from 5.9 to 5.6 on every seed.
- `philanthropy.datasets.load_karlan_list(path)` reads `AERtables1-5.dta` from
  the Karlan and List (2007) matching-grant experiment (openICPSR 113224; data
  CC BY 4.0, copyright American Economic Association 2007) from a local path:
  50,083 prior donors to one charity, one 2005 letter, with the matching-grant
  offer, match ratio, cap and example ask randomised.
- Results pages: Karlan and List as a fifth real file
  (`--karlan-list-path` on both results scripts). The response model beats the
  best of four simple rules on it (top 10%: 8.0 vs 5.5 of 100 gave), a second
  organisation after KDD Cup 1998, so the Response bottom line moves from "Use
  the model (tested on one organisation so far)" to "Use the model". The
  amount model is about the same as the donor's largest past gift. The first
  `UpliftTLearner` rows (who the matching-grant offer moves most) are about the
  same as ranking by recency. Split, features and rule sets were fixed before
  the run, one configuration each.
- `philanthropy.models.suggest_ask(last_gift, avg_gift, stretch=0.10, round_to=25)`:
  the simple ask rule as one function call (the larger of last and average
  gift, raised 10% and rounded up to the next $25). With `stretch=0` and
  `round_to=None` it is the rule the Results pages test; on no real file
  does `AskAmountRecommender` land within 25% of the next gift reliably more
  often than it.
- `AskAmountRecommender` checks itself against that rule when `last_gift_idx`
  and `avg_gift_idx` are set: `fit` scores a model trained on 80% of the rows
  and the rule on the other 20%, and sets `beats_rule_`, `rule_mae_` and
  `model_mae_`. The returned model is still fit on every row, so predictions
  are unchanged. With fewer than 100 rows or no indices the three are `None`.
- `philanthropy.utils.check_label_floor(y, task)` returns `"run"` or
  `"not enough labels"` for `task="lapse"` or `"major_gift"`, so a hosted
  no-code page and a Python user apply the same cutoff before fitting. The
  floors (`LABEL_FLOORS`: 100 in the rarer class for lapse, 800 for major
  gift) come from validation-fold learning curves on DonorsChoose and KDD
  Cup 1998, ten seeds each; the test splits were not used. Below the floor,
  at least one training draw in ten scored below the simple rule on held-out
  data (lapse), or most did (major gift).

- Results, who to mail: each tab gains a table of the same fit at four
  costs per letter ($0.50, $0.68, $1.00, $2.00), with letters sent, net
  revenue against mailing everyone, and a paired bootstrap range on the
  difference. Generated from `results.json` and covered by the drift test.

### Changed
- The pre-push hook (`scripts/install_hooks.sh`) now runs only the test
  collection check and flake8, which take seconds, instead of the full suite.
  CI already runs the full suite on every push to a PR. Contributors run
  `make ci` once before opening the PR rather than before every push; rerun
  `sh scripts/install_hooks.sh` to pick up the new hook.
- CI measures coverage on one test leg (Ubuntu, Python 3.13) instead of all
  six. The other legs run the same tests without tracing; on the older
  Pythons tracing roughly doubled the suite's time. Both coverage floors
  are unchanged.
- KDD Cup 1998 feature checks, recorded with no published number moving.
  On the 15% validation fold (paired bootstrap, seed 42), adding the 12
  donor-table columns and three as-of mailing features (`promos_received`,
  `response_rate`, `months_since_last_promo`) to the response models did
  not lift top-10%: `DonorPropensityModel` 9.36 to 9.64 [-0.91, +1.47],
  `MajorGiftClassifier` 9.71 to 9.29 [-1.82, +0.77], so both keep the four
  RFM columns (cup98VAL, looked at during exploration, showed
  `DonorPropensityModel` +0.74 [+0.28, +1.12]; validation decides). The
  same mailing features in the who-to-mail response model lowered
  validation net revenue on all five seeds. Isotonic calibration of
  `DonorPropensityModel` is not recommended (docstring note). KDD98 lapse
  stays on its promotion panel: only 242 donors gave in the last history
  promotion against 4,843 to the 97NK mailing, so the shared builder's
  "gave in T" population is not real there; the parity test's skip reason
  says so.
- Results, ask: the page is retitled "What will this donor give next?" and
  opens with the bottom line and the `suggest_ask` call, since a suggested
  ask is a policy choice built on that forecast. A new row on every ask
  bench reports the share of next-gift dollars from the top 10% of donors
  ranked by the model and by the rule: about the same on KDD Cup 1998,
  DonorsChoose and Karlan and List, and slightly ahead for the model on PSID
  in every test wave (43 vs 42 of every $100). The ask bottom line is
  unchanged: use the simple rule.
- `MajorGiftClassifier` class weights for the upgrade model, checked on the
  PSID wave 2013 and DonorsChoose FY2014 validation folds (five seeds) and
  not adopted: `"balanced"` lowered DonorsChoose top-1% from 22.9 to 21.8
  of 100, and `{0: 1, 1: 5}` lowered DonorsChoose top-10% from 5.9 to 5.8.
  `score_leadership_prospects` keeps the unweighted default.
- Results, lapse: the DonorsChoose and PSID lapse benchmarks now build their
  rows with `build_snapshots` / `period_snapshots`, so both feed
  `LapsePredictor` the same shared gift columns (a new parity test checks
  this), and both ask the question only of donors with at least two years
  (PSID: waves) of giving, the donors a retention program can act on.
  DonorsChoose: base lapse rate 84 to 64 of 100; top-10% lapse hit rate
  81 vs 81 for the rule (was 88 vs 90), so its index cell now shows the
  lapse read, "about the same", instead of the retention read; the
  retention read still beats the rule, 75 vs 68 (was 50 vs 45). PSID: base
  rate 26 to 23; top-10% 55 vs 47 (was 59 vs 54), still beats the rule.
  The lapse bottom line is unchanged: use the model. The how-to-read worked
  example uses the new DonorsChoose retention numbers.
- Results: every verdict now comes from the data instead of a fixed 15%
  margin. The model beats the rule only when the paired model-minus-rule gap
  stays on the model's side across bootstrap redraws of the same test donors
  (one split) or in every seed and test year (walk-forward), loses when it
  stays on the rule's side, and is "about the same" otherwise. A single split
  with no bootstrap interval gets no verdict. The index table is generated
  from `results.json`, drops the sample-data column, and ends each row with
  one of six bottom lines by a fixed rule: "Use the model" needs wins from two
  independent sources (two organisations, or one file in every one of several
  test years) and no counted loss, and wins from one organisation in single
  splits read "Use the model (tested on one organisation so far)", kept apart
  from "Use the model for your top slice only"; KDD Cup 1998 and cup98VAL
  count as one source, the KDD98 $50 upgrade proxy is shown but not counted,
  and sample data is a code check only. The table also reports the
  who-keeps-giving list with its lift over random where more than 80 in 100
  donors lapse. The scoreboard plots each real-file result as a multiple of
  the rule with its range, and a drift test ties each hand-written number on
  the Results pages to its `results.json` key and fails when one drifts. The
  PSID lapse comparison adds "this-wave total (negated)" as a third simple
  rule.
- `PlannedGivingIntentScorer` now boosts with `HistGradientBoostingClassifier`
  (`max_iter=n_estimators`, `max_depth=3`, the old backend's depth) and
  accepts missing values, so blank age or wealth-screening columns need no
  imputation. On the sample-data response stand-in (5 seeds) the old
  backend scored mean ROC-AUC 0.699 and top-10% 75 of 100; the new one
  0.700 and 76. HGB at its default depth scored 0.692 and 73, so the depth
  is pinned. Two configurations were compared.
- `LABEL_FLOORS` re-measured with the top-10% hit rate next to ROC-AUC, on
  validation folds outside every published test fold (the lapse floor was
  first read on DonorsChoose FY2017, which is one of the Lapse page's test
  years). Both floors stand. Lapse, PSID wave 2013: from 100 lapsed
  households every draw beat the best rule on both measures. On
  DonorsChoose FY2014 the model needs 200 retained donors to clear the rule
  on ROC-AUC every time and only ties it at the top 10% at any count, as the
  Lapse page says. Major gift, KDD Cup 1998 validation fold: at 800
  positives `DonorPropensityModel` beat the rule in 10 of 10 draws on both
  measures and `MajorGiftClassifier` in 10 of 10 on ROC-AUC and 9 of 10 at
  the top 10%. On PSID's real $1,000 upgrade label (validate wave 2013),
  `MajorGiftClassifier` beat the rule at the top 10% in every draw from 800
  positives (39 vs 35 of 100) and in 6 of 10 at 400.

### Fixed
- `check_label_floor` raises a `ValueError` on missing (NaN or None) labels
  instead of counting them as a class, which let a file of non-lapses and
  blank outcomes pass the lapse floor with zero lapses.

### Removed
- `scripts/issue-drafts/_DISCUSSION_who_is_using_this.md`: the draft was
  posted as Discussion #158 on 2026-09-05, and its "zero dependents" and
  "zero usage" lines are now out of date. The Discussion itself is the record.

### Documentation
- Results: a "How to read these pages" page (the simple rule per question,
  "random", list lengths with a 10,000-donor worked example from the
  DonorsChoose retention read, "about the same", and the five bottom lines)
  and a "The datasets behind these pages" page with one card per file (what it
  is, who gave to whom and when, how big, what it can and cannot test, terms),
  including the Karlan and List experiment that is on hand but not yet used.
- `MajorGiftClassifier` docstring: states the calibration method (sigmoid,
  5 folds) and that boosting already stops early above 10,000 rows, and
  records two tuning checks on a KDD Cup 1998 validation fold (five seeds):
  `learning_rate=0.05, max_iter=300` scored the same as the defaults, and the
  textbook RFM monotonic constraint lowered ROC-AUC (0.601 to 0.595). No
  default changed. A `DonorPropensityModel` leaf-fraction check on the same
  fold (0.01, 0.02, 0.05) also stayed inside the 0.008 default's seed range,
  so that default stays too.

### Added
- Results pages now show PSID (Panel Study of Income Dynamics) household
  survey results for leadership upgrade, lapse (both the lapse read and the
  "who gives again" retention read) and suggested ask, with a chart, verdict
  and "What the model looks at" tab each, tested on survey waves 2015 to 2021
  in turn. On PSID the upgrade model beats the best simple rule at the top
  10% (41 vs 32 in 100) and so does the lapse model (59 vs 44 in 100); the
  retention read and suggested ask come out about the same as the rule. The
  scoreboard and the models-by-datasets table now fill the PSID column, and
  leadership upgrade and lapse read "depends on your data": each now has one
  real win next to a real loss or tie, short of the two real wins "beats the
  rule" requires. PSID data are not redistributed; every PSID number is an
  aggregate, regenerated from a user's own extract with `--psid-data` /
  `--psid-do`. The DonorsChoose and PSID charts and driver tabs share one
  code path in `scripts/make_results_pages.py`. The scoreboard legend is
  spaced tighter so all six entries fit now that PSID has a marker, and its
  markers no longer draw a stray white line in the dark theme.
- `scripts/make_results_pages.py --with-momentum`: runs each synthetic model
  (and KDD98's upgrade model, with `--with-kdd98`) a second time with
  `include_momentum=True`, writing a `<key>_momentum` entry next to the
  existing one in `results.json` (five-seed range included for synthetic
  rows, a plain-language `method` string for the docs to quote). Every
  existing key and default number is untouched; verified byte-identical
  except `_env.git_sha`. On the sample panel, momentum moves the upgrade
  model's top-10% hit rate from 21.6% to 25.4%, inside the five-seed noise
  band rather than clearly outside it; on KDD98's real gift history it makes
  no difference at all (verified: 90.9% of the KDD98 momentum values are
  real, non-NaN numbers with genuine spread, but `donor_feature_importance`
  gives momentum, and several of the model's existing columns, exactly 0.0
  permutation importance on that file). Response, lapse and ask get the
  same momentum columns applied to this script's own per-donor annual
  panel (not the shipped `RFMTransformer`/`activities_to_features` path);
  none of their verdicts change either.
- `philanthropy.utils.trailing_slope_features`, a shared as-of trailing-window
  OLS slope/relative-slope helper over annual (or other evenly-spaced) bins
  of a donor time series. `RFMTransformer(include_momentum=True)` and
  `activities_to_features(..., include_momentum=True)` both opt in to add
  slope columns for their base series (giving total, gift count, largest
  gift; activity count/hours/amount), and `build_leadership_snapshots`/
  `score_leadership_prospects(..., include_momentum=True)` add the same slopes
  plus `fy_total_growth_ratio`. All three default to `False`: the feature
  is new, additive, and off by default so it does not change any shipped
  estimator's default training features ahead of the JOSS submission
  freeze. Leakage-safe: every window is anchored at the caller's own as-of
  cutoff and a window with fewer than two observed periods returns `NaN`
  rather than fabricating a trend from one point.
- `scripts/benchmark_models_vs_baselines.py` and `scripts/make_results_pages.py`
  gain opt-in `--donorschoose-path` / `--psid-data` + `--psid-do` flags:
  upgrade, lapse (plus a retention read), and ask vs the same rule sets
  already used for the synthetic panel and KDD98, on real DonorsChoose
  (ICPSR 37898) and PSID giving history, with and without momentum
  features. CI never needs either file; both sections are skipped entirely
  when the paths are not given. Response, who-to-mail, and planned-giving
  have no mailing/appeal or bequest data in either file, so those sections
  are left out with a one-line reason instead. Aggregate numbers and fold
  metadata (subsample fraction and seed, fold years/waves, n per fold,
  base rate) only; no donor- or household-level rows anywhere.
- Results pages now show DonorsChoose on the leadership upgrade, lapse, and
  suggested-ask pages: a headline sentence, a top-1/5/10% chart, a verdict,
  and a "What the model looks at" tab, alongside the existing sample-data
  and KDD98 tabs. Lapse covers both the plain lapse read and the "who gives
  again" retention read, with the ~84% base rate spelled out in plain
  words. Response, who-to-mail, and planned giving each gain a one-line
  "not testable on DonorsChoose or PSID" note quoting the actual reason
  (`results.json`'s `*_donorschoose_note`/`*_psid_note` strings). Every page
  with a momentum variant now states, in prose, whether adding this year's
  giving trend as its own feature moved the top-10% hit rate outside the
  normal run-to-run range, including the KDD98 upgrade case where the
  numbers are identical with and without it (the model gives those columns
  zero weight on that file, not "no data"). `docs/results/index.md` and the
  home page's scoreboard chart now show a models x datasets matrix (Sample
  data, KDD Cup 1998, cup98VAL, DonorsChoose, PSID), using "can't test
  (reason)" where a file structurally cannot answer a question and
  "pending: data not on hand" for PSID, which is wired into the benchmark
  script but not yet run in this environment. `scripts/make_results_pages.py`
  gains `upgrade_drivers_donorschoose`/`lapse_drivers_donorschoose`/
  `ask_drivers_donorschoose` and a `"donorschoose"` entry in
  `DATASET_GROUPS_AVAILABLE` (giving history and recency only: the
  DonorsChoose Donations file has no wealth, demographic, or mailing-history
  columns at all).

### Fixed
- `scripts/benchmark_models_vs_baselines.py`'s synthetic `$1K upgrade` bench
  (`bench_upgrade`) trained `MajorGiftClassifier` on every numeric snapshot
  column including the raw `fiscal_year` label, unlike the KDD98 upgrade
  bench and `score_leadership_prospects`, both of which already drop it. The
  model could partly key on which split a row came from rather than who
  upgrades. `fiscal_year` is now excluded there too. Regenerated
  `docs/assets/results/results.json` and the upgrade results page: the
  synthetic $1K upgrade top-10% hit rate moves from 24.9% to 21.6% model
  (rule stays 17.7%, random 12.5%); still a win, just a smaller one. The
  KDD98 upgrade numbers are unchanged, since that bench already excluded
  `fiscal_year`.
- Results-page charts (`docs/results/*.md`, the homepage, Start Here) were
  baked as light-background PNGs only, so on the site's default dark theme
  each one rendered as a bright rectangle. `scripts/make_results_pages.py`
  now renders a matching dark variant of every chart, and each page embeds
  both via Material's `#only-light`/`#only-dark` image switch. Also fixes a
  reference-line label overlapping the first bar on two charts, and widens
  the docs layout's content column on pages with both a nav and
  table-of-contents sidebar (previously squeezed on wide screens).
- `FiscalYearTransformer` and `EncounterRecencyTransformer` labelled every
  January-start fiscal year one year ahead of the calendar year (e.g. June
  2024 came out as FY2025 instead of FY2024). Every other start month was
  already correct. **Behaviour change** for `fiscal_year_start=1` users: the
  fiscal year now equals the calendar year, matching the class docs. The
  same off-by-one existed in `score_leadership_prospects`'s current-fiscal-year
  calculation and in `build_leadership_snapshots`'s `_fy_end` cutoff; both now
  share the corrected, vectorised `philanthropy.utils._validation.
  fiscal_year_and_quarter`/`fiscal_year_for` helpers instead of duplicating
  the formula (and a per-row `.apply`) in four places.
- `UpliftTLearner.fit` now rejects a non-``{0, 1}`` `y` instead of silently
  mis-scoring: `_prob_give` resolves the positive class as the literal
  integer `1`, so a string-labelled ("yes"/"no") arm that saw only one class
  during fit produced a sign-flipped uplift score with no error.
- `WealthScreeningImputerKNN(strategy="knn")` filled every column via
  `KNNImputer.transform`, not just the columns in `wealth_cols`/
  `imputed_cols_`, contradicting the documented "subset of columns to
  impute". It still fits `KNNImputer` on the whole matrix (neighbour
  distance benefits from every column), but now only writes back the
  columns it is contracted to impute.
- `GratefulPatientFeaturizer` counted a missing `service_line` value as a
  distinct line: `.astype(str)` turned `NaN`/`None` into the literal string
  `"nan"`, which could be counted in `distinct_service_lines` or win
  `primary_service_line`. Missing values are now excluded before counting.
- `FiscalYearGroupedSplitter` counted a `NaN` fiscal year as one more
  distinct year in `get_n_splits`, while `split` silently produced one fewer
  fold (every comparison against `NaN` is `False`, so those rows never
  landed in a train or test fold). `get_n_splits` and `split` now agree, and
  a `UserWarning` names how many rows were excluded.
- `gift_concentration_gini`/`top_donor_share` returned `NaN` (with a
  `RuntimeWarning`) for an infinite gift amount, despite `_clean_nonneg_
  amounts` documenting that it keeps only finite values. Both now raise
  `ValueError`, matching the existing negative-amount check.

### Added
- `philanthropy.datasets.load_psid_philanthropy`: reads a user-downloaded
  PSID (Panel Study of Income Dynamics) individual-level cross-year extract
  into a long household giving/volunteering table (`household_key`, `year`,
  `total_giving`, per-category giving, family income, wealth, and three
  separate volunteering-hours measures: annual (head/spouse split), asked
  in 2001; a household-level "regularly volunteered" total, asked in 2003
  and 2005; and typical week (head/spouse split), asked 2017 onward; these
  are never combined into one series).
  Never downloads or redistributes PSID data; the file stays on the user's
  own machine, fetched under their own PSID Data Center registration.
- `philanthropy.datasets.load_donorschoose`: reads a user-downloaded
  DonorsChoose Open Data Donations file (ICPSR 37898, DS0001) into a gift
  table (`donor_id`, `gift_date`, `gift_amount`, plus `donor_type` and three
  payment-source flags). `gift_date` is month-resolution only, matching the
  source's `CREATED_MONTH` field. It never downloads or redistributes the
  dataset; the file stays on the user's own machine, fetched under their own
  ICPSR account.
- `philanthropy.metrics.demographic_parity_difference`: `max(selection_rate)
  - min(selection_rate)` across protected groups, alongside the existing
  `disparate_impact_ratio`. The ratio is noisy when rates are small (0.01
  vs 0.02 gives a ratio of 0.5 but a difference of 0.01); report both.
- Every Results page now has a second tab group, "What the model looks at",
  one tab per dataset: a plain-language table of the feature groups the
  model is given, its top 5 drivers with direction (raises/lowers/depends
  the score), and, where a feature-set comparison exists, what adding more
  signal did to the top-N hit rate. `scripts/make_results_pages.py` computes
  driver size with `philanthropy.inspection.donor_feature_importance`
  (permutation importance) and direction from
  `sklearn.inspection.partial_dependence`, and writes one snippet per
  (model, dataset) under `docs/results/_features/`, included into each page
  with `pymdownx.snippets`. Raw column names, importance intervals and
  method details stay in a collapsed "For analysts" note.

### Changed
- Renamed the unreleased upgrade model before its first release, so the name
  says what it predicts (a mid-level donor reaching the leadership-giving
  level, $1,000 by default): `score_leadership_prospects` is now
  `score_leadership_prospects`, `build_leadership_snapshots` is now
  `build_leadership_snapshots`, and `philanthropy train --task upgrade` is now
  `--task leadership`. The Results page moved from `results/upgrade/` to
  `results/leadership/`; the old address keeps a stub linking to the new one.
  No deprecation aliases, since none of these shipped in 0.8.0.
- `scripts/benchmark_models_vs_baselines.py` now checks `GiftIntervalCalibrator`
  coverage over 10 seeds at 80/90/95% requested levels instead of 5 seeds at
  90% only. Attained coverage is 0.784/0.9045/0.9539, all within 2 points of
  target, confirming the earlier single-seed 88% vs 90% reading was noise.
- `scripts/benchmark_models_vs_baselines.py` adds a second synthetic column
  (`synthetic_panel_full_features`) that re-runs `DonorPropensityModel`,
  `MajorGiftClassifier` and `LapsePredictor` on the full 8 as-of columns
  instead of the 3 the models were fed while the baseline rules already saw
  5+. `LapsePredictor` moves from losing everywhere (AUC 0.638) to beating
  the rule on synthetic (AUC 0.680, top-10% 94.4% vs 83.1%); response models
  move only slightly. The same feature set hurts lapse on KDD98 (below
  chance, AUC 0.488 in an earlier probe), so this is reported next to the
  3-feature row, not adopted as a default.

### Added
- `AskAmountRecommender(target_mode="relative")`: fits
  `log(y / max(last_gift, avg_gift))` and multiplies the prediction back out,
  via new `last_gift_idx`/`avg_gift_idx` parameters (E.12d). Evaluated under
  E.11a against an "absolute + max(last, avg) as a feature" candidate: the
  feature-only candidate wins the validation fold and still loses the KDD98
  test (MAE 4.077 vs the rule's 3.875) and cup98val (4.147 vs 3.866), so the
  default stays `target_mode="absolute"` without the extra feature; the
  option ships for anyone who wants to try it on their own file.
- `LapsePredictor(backend="hist_gradient_boosting")`: an opt-in
  `HistGradientBoostingClassifier` backend alongside the default
  `RandomForestClassifier` (E.12d). Picked against the default on cup98val
  (to avoid KDD98's own near-empty validation period, E.12c), the two backends
  tied exactly (AUC 0.555 vs 0.555), so the default stays
  `backend="random_forest"`.
- `LapsePredictor.predict_retention_score` and the `retention_read_` fitted
  attribute (`True` when the training lapse rate exceeds 80%): on a file
  where almost everyone lapses, ranking by lapse score barely beats random,
  but the 10% *least* likely to lapse still beats the rule on both KDD98
  files (top-10% retention 6.5% vs 5.6%, top-5% 7.6% vs 6.0%).
- `philanthropy.ingest.donorperfect_gifts_to_features` /
  `read_donorperfect_gifts`: a bridge from a DonorPerfect gift export to the
  donor-level feature table. DonorPerfect's commitment/split-total signal
  lives in `record_type`, not its own `gift_type` field (a payment-method
  descriptor); `P` (Pledge) and `M` (a split gift's Main total) are excluded
  by default via `DEFAULT_EXCLUDED_RECORD_TYPES`, keeping `G` (a regular
  gift, a pledge payment, or a split entry). Registered as `"donorperfect"`
  in `read_gifts`/`GIFT_SOURCES`. Field names and the `record_type`
  vocabulary are taken from SofterWare's DonorPerfect Online XML API
  Documentation (dp_savegift's `@record_type` parameter and its "Split
  Gifts"/"Pledge Notes" sections).
- `philanthropy.ingest.bloomerang_transactions_to_features` /
  `read_bloomerang_transactions`: a bridge from a Bloomerang transaction
  export to the donor-level feature table, following the same
  commitment-versus-payment shape as the Raiser's Edge and NPSP bridges.
  `Pledge` (the up-front commitment) and `Recurring Donation` (the
  not-yet-charged schedule) are excluded by default via
  `DEFAULT_EXCLUDED_ENTRY_TYPES`; `Donation`, `PledgePayment` and
  `RecurringDonationPayment` are kept. Registered as `"bloomerang"` in
  `read_gifts`/`GIFT_SOURCES`. Field names and the entry-type vocabulary are
  taken from Bloomerang's REST API V1 docs
  (https://bloomerang.com/api/rest-api-v1/) and Help Center Transactions
  Report article
  (https://help.bloomerang.com/en/articles/13382625-transactions-report).
- `scripts/benchmark_models_vs_baselines.py --with-blood`: an opt-in second
  real dataset, the UCI Blood Transfusion Service Center file (Yeh, Yang and
  Ting 2009, DOI 10.24432/C5GS39, CC BY 4.0; 748 repeat blood donors,
  ~12 KB, cached in `~/philanthropy_data`). `bench_blood` scores response
  and lapse on five stratified 70/30 splits against the same rule sets
  used elsewhere. Blood, not money, so it only tests ranking of repeat
  donors. Results are noisy (22 donors in each top 10%): `DonorPropensityModel`
  top 10% 64.6% [54.5-72.7] vs 58.2% for the RFM cell score (random 23.8%),
  AUC 0.745 vs 0.692; `MajorGiftClassifier` 55.5% vs 58.2% (AUC 0.725);
  `LapsePredictor` ties months since last donation (91.8% each, AUC 0.695
  vs 0.697). Benchmark-only; no loader is added to the package.
- `scripts/benchmark_models_vs_baselines.py` gains `bench_forecast`, the
  first benchmark of `FinancialForecastModel`: monthly giving totals on the
  synthetic panel, last 12 months held out, against a 12-month mean and
  seasonal naive. `predict_revenue_forecast` loses to the 12-month mean (5
  seeds: 13.0% error on the held-out year's total vs 7.2%; monthly MAPE
  16.9% vs 13.0%), because its roll-forward never sees the future months'
  features. `predict` on the future months' calendar features wins (3.5%
  and 11.3%). The `predict_revenue_forecast` docstring now says when to use
  `predict` instead. Sample data only; no real monthly revenue series has
  been tested yet.
- `scripts/benchmark_models_vs_baselines.py --with-cup98val` now also scores
  the response (`DonorPropensityModel`, `MajorGiftClassifier`), lapse and
  ask models on KDD Cup 1998's own held-out validation file
  (`bench_kdd_val_models`), each fit exactly as in its learning-file row.
  On that file `MajorGiftClassifier`'s top 10% found 8.9% responders
  against 7.4% for the best rule (RFM cell score) and 5.1% at random;
  `LapsePredictor` and `AskAmountRecommender` still do not beat their
  rules. `make_results_pages.py --with-cup98val` writes the response numbers
  into `results.json`, and the Response page and Results index now report
  the real-file result next to the sample-data one.
- `MajorGiftClassifier` gains `monotonic_cst`, `min_samples_leaf`, and
  `max_leaf_nodes` passthrough parameters to the underlying
  `HistGradientBoostingClassifier`, alongside the existing `max_iter` and
  `learning_rate`. Purely additive: defaults match the underlying
  estimator's own defaults, so existing code is unaffected.
- `philanthropy.datasets.fetch_kdd98_val_donors`: fetches KDD Cup 1998's own
  held-out validation file (`cup98VAL` + its `valtargt` answer key), a
  second 96,367-donor file that was never part of the learning file
  `fetch_kdd98_donors` returns and never touched by that file's own
  55/15/30 split. `scripts/benchmark_models_vs_baselines.py` gains
  `bench_kdd_cost_aware_val` and a `--with-cup98val` flag (opt-in, a second
  ~37MB download): it fits the same cost-aware mail-selection models on the
  learning file's own train split and scores them on this genuinely
  held-out file instead. Reported next to the existing random-split number,
  not replacing it: net revenue $13,764 vs $10,560 for mailing everyone
  (ROI 0.32 vs 0.16), the same shape as the learning-file split's $4,542 vs
  $3,149. `scripts/make_results_pages.py --with-cup98val` writes this into
  `docs/assets/results/results.json` and the "Who to mail" page now states
  both numbers.
- Test covering `PropensityScorer.predict_proba` when `fit` saw only one
  class (closes #50).
- A "Results" section in the docs (`docs/results/`): one page per model
  ($1K upgrade, response/major gift, lapse, suggested ask, planned giving,
  who to mail) written for fundraisers, not data scientists, comparing each
  model's picks against a simple rule and against random in plain donor
  counts, plus an index page with a one-line verdict per model.
  `scripts/make_results_pages.py` generates every number and chart these
  pages cite (`docs/assets/results/`), by running
  `scripts/benchmark_models_vs_baselines.py` and one worked example of
  `score_leadership_prospects`, so nothing on the pages is typed by hand.
- `scripts/benchmark_models_vs_baselines.py`: pairs every estimator with the simple domain rule it is meant to replace (rank by last year's total, predict last gift, mail everyone) and evaluates both on held-out, walk-forward splits: top-1%/5%/10% hit rate and lift, ROC-AUC, average precision and decile calibration for classifiers; MAE and within-25% for amount predictions; net revenue/ROI for cost-aware mail selection. Runs on the synthetic donor panel (five seeds, mean and min-max) by default; `--skip-kdd98` stays fully offline, `--fast` gives a one-seed smoke run, and `--out` writes the results table to JSON and CSV. Separate from the existing `scripts/benchmark_models.py` per-model accuracy table, which this does not replace or touch.
- `philanthropy.ingest.map_columns(df, mapping, *, required=...)`: renames a
  user-supplied export's headers to canonical names and raises one
  `ValueError` listing every still-missing required column, for callers
  building a column-mapping UI over an arbitrary CRM export.
- `philanthropy.ingest.read_npsp_opportunities` and
  `npsp_opportunities_to_features`: a Salesforce Nonprofit Success Pack (NPSP)
  Opportunity export bridge, alongside the existing CiviCRM and Raiser's Edge
  ones. Counts only closed/won stages by default (`Closed Won`,
  `Awarded`, `Posted`, via `DEFAULT_INCLUDED_STAGES`) so a `Pledged`
  instalment, an open pipeline stage, and `Closed Lost` are not summed as
  gifts. Wired into the CLI as `philanthropy features --source npsp`.
- `philanthropy.ingest.activities_to_features(activities, *, as_of,
  donors=None)`: aggregates a long, multi-source activity log (event
  attendance, volunteer shifts, email clicks, ...) into per-donor,
  per-activity-type engagement features (`<type>_count_12m`,
  `<type>_count_36m`, `<type>_days_since_last`, `<type>_distinct`, plus
  `<type>_hours_12m` / `<type>_amount_12m` when those columns are present),
  cut at `as_of` so nothing dated after the cutoff is counted. A new activity
  type never needs new model code; it just yields its own columns.
- `philanthropy.ingest.read_gifts(path_or_df, *, source=...)`: reads (if given
  a path) and aggregates a gift export in one call, looking up the CiviCRM,
  Raiser's Edge or NPSP reader-and-aggregator pair by name from the new
  `GIFT_SOURCES` registry, for a caller working with more than one CRM export
  format. Keyword arguments other than `source` pass straight through to the
  matched aggregator, so `include_stages`, `exclude_gift_types` and `statuses`
  all still work.
- `philanthropy.ingest.build_leadership_snapshots(gifts, *, fiscal_years,
  threshold=1000, band=(100, 999), fiscal_year_start=7, activities=None,
  donors=None)`: builds a per-donor, per-fiscal-year training table for an
  upgrade model, one row per donor whose fiscal-year-T giving lands in the
  upgrade band, with gift-derived features (prior-year totals, trend,
  largest gift, gift count, consecutive years given, months since last
  gift), optional joined activity and donor-attribute columns, and a
  `target` reading whether the donor crossed `threshold` in fiscal year T+1.
  Everything but `target` is computed from data through the end of T; the
  output's `fiscal_year` and donor-id columns feed directly into
  `FiscalYearGroupedSplitter`.
- `philanthropy.models.score_leadership_prospects(gifts, *, activities=None,
  donors=None, threshold=1000.0, band=(100.0, 999.0), fiscal_year_start=7,
  as_of=None, top_n=None, baseline_giving_threshold=None, random_state=None)`:
  the fit-and-score entry point over `build_leadership_snapshots`. Trains a
  `MajorGiftClassifier` on every fully-resolved historical fiscal year
  (excluding `fiscal_year` itself from the feature set), validated with a
  walk-forward `FiscalYearGroupedSplitter` fold, then scores today's
  band-qualifying donors (cut at `as_of`, never at a future fiscal-year end)
  with a model refit on all history. Returns a `(scores, report)` pair:
  `scores` has `affinity_score`, `rank`, `decile`, a per-donor `top_reasons`
  heuristic built from global permutation importance, and a `suggested_ask`
  left `NaN` (no ask-amount label exists yet to train one honestly);
  `report` carries training-row counts, a low-data warning under ~500 rows,
  the `activities_to_features` id-match warning, a per-decile breakdown of
  the held-out fold (`deciles`), `roc_auc`/`average_precision`, and two
  named-baseline upgrade rates and lifts ("gave >= X last FY" and "top N by
  FY total", `top_n` defaulting to ~10% of the fold). Wired into the CLI as
  `philanthropy train --task leadership`; `philanthropy features` gained
  repeated `--activity TYPE=PATH` and `--as-of` flags to fold engagement
  data into the feature table the same way.
- `examples/notebooks/05_leadership_upgrade.ipynb`: a leadership annual-giving
  upgrade model, `build_leadership_snapshots` plus a synthetic activity log into
  `MajorGiftClassifier`, validated with a fiscal-year walk-forward split and
  compared against a naive "gave $500+ last FY" rule on top-N upgrade rate.
- `philanthropy validate` now reports average precision and a 10-row decile
  table (n, positives, hit rate, lift over the base rate), plus a `--top-n`
  hit-rate/capture line (count or percentage, default 10% of rows). The
  existing precision/recall/F1 are now labelled "at threshold 0.5" so they
  aren't mistaken for the whole picture on a rare, imbalanced target, where
  they can look broken even when the model ranks donors well.

### Changed
- Logo: the heart-and-arrow mark is replaced by a phi (φ) with a dot above
  it. φ is the "phil" (love) in philanthropy and also the golden ratio; the
  dot is the gift, or the score, rising out of it. Same three places as
  before: `overrides/.icons/philanthropy/phi.svg` (renamed from
  `heart-rise.svg`) is the header logo, `docs/assets/logo.svg` is the
  favicon, and `docs/assets/logo.png` is the README lockup, with the
  wordmark lettering unchanged.
- `AskAmountRecommender(loss=...)`: a new parameter passed through to the
  backend `HistGradientBoostingRegressor`, default changed from
  `"squared_error"` to `"absolute_error"`. Ask amounts are right-skewed (a
  few big gifts, many small ones), and squared error was fitting the
  conditional mean, which overshoots most donors; absolute error fits the
  median instead. Chosen on the KDD Cup 1998 validation split (55/15/30):
  absolute error's validation MAE ($4.19) beat squared error's ($4.72), and
  on the held-out test split absolute error's MAE ($4.11, 65 of every 100
  suggestions within 25% of the actual gift) beats the old default's ($4.56,
  59 of 100). It still does not beat the best ask rule on this file (max of
  last gift and average gift, $3.88 MAE, 67 of 100), so the Results page
  keeps its "use the rule instead" verdict, just with the updated numbers.
  The "Who to mail" cost-aware selection multiplies response probability by
  expected gift size, which needs the conditional mean, so the benchmark
  now passes `loss="squared_error"` there explicitly and that page's numbers
  are unchanged. `docs/results/ask.md` and `docs/assets/results/results.json`
  are regenerated in this PR.
- `scripts/benchmark_models_vs_baselines.py` now pairs every model with a
  fixed set of 2-4 named baseline rules (e.g. lapse: LYBUNT/SYBUNT flag,
  years since last gift, shortest giving streak, gave nothing last period)
  instead of one hand-picked rule, and reports the verdict against whichever
  rule in that set is toughest. KDD Cup 1998 moves from a 70/30 split to
  55/15/30 (train/validation/test; validation is unused until a later PR
  adds hyperparameter choices), and every KDD98 row now carries a bootstrap
  95% interval on its top-10% hit rate and ROC-AUC. Some verdicts flip now
  that the comparison rule is honestly the strongest one available:
  `MajorGiftClassifier` on the response task goes from "about the same" to
  losing on synthetic data (still wins on KDD98); `LapsePredictor` loses on
  both synthetic and KDD98 once "years since last gift" is in the rule set;
  `AskAmountRecommender` now loses on synthetic data too, not just KDD98,
  once "max(last gift, average gift)" is in the rule set. The "Who to mail"
  page's net revenue moves from $4,240 to $4,542 as a side effect of the
  KDD98 split change (still beats mailing everyone's $3,149). All affected
  Results pages and `docs/assets/results/results.json` are regenerated in
  this PR.
- `DonorPropensityModel`'s default `min_samples_leaf` goes from `1` to
  `0.008`, a fraction of the training rows (so about 2 rows per leaf on a
  250-donor file and 400 on 50,000). With single-sample leaves and no depth
  limit the forest memorised its training rows, so `predict_proba`
  collapsed to near-0/1 votes and the ranking was mostly noise. A fraction
  rather than a fixed count, because a fixed 200 left files under about 400
  rows with no possible split (every donor got the same score) and squeezed
  mid-sized files' affinity scores into roughly 15-61. 0.008 was picked on
  the KDD98 validation fold out of 0.001/0.002/0.004/0.008 (4
  configurations) and won on a second split seed as well; the test split
  was scored once, after the choice. On the KDD98 test split, against the
  best of its three response rules (RFM cell score): top-1% hit rate 6.3%
  before, 11.5% after (rule 9.1%); top-5% 6.9% to 10.6% (rule 7.1%);
  top-10% 6.4% to 8.8%, 95% interval 7.8-9.9% (rule 7.5%); ROC-AUC 0.511 to
  0.593, interval 0.578-0.607 (rule 0.567). On the synthetic panel (5 seeds)
  ROC-AUC goes from 0.638 to 0.712 and top-10% from 61.5% to 76.5%, but it
  still trails both the lifetime-giving rule (79.2%) and
  `MajorGiftClassifier` (77.1%) there, so `MajorGiftClassifier` stays the
  recommended response model. In `scripts/benchmark_models.py`'s accuracy
  table its ROC-AUC moves from 0.810 to 0.841; the golden file and
  `docs/explanation/benchmarks.md` are updated. Pass `min_samples_leaf=1`
  to get the old behaviour. The model now also accepts missing values
  (`NaN`) directly, as `RandomForestClassifier` has since scikit-learn 1.4.

### Fixed
- `scripts/benchmark_models_vs_baselines.py`: the KDD98 response rule "RFA_2
  frequency then last gift" mapped the string codes `"1"`, `"2"`, `"5"`, but
  `RFA_2F` loads as the integers 1 to 4, so every donor mapped to 0 and the
  rule was really just "last gift". It now ranks on `RFA_2F` directly. This
  makes the rule stronger on the learning-file split (top 10% 8.3% instead
  of 7.5% for the previous best rule), so `MajorGiftClassifier`'s top-10%
  edge there (8.9%) is now inside the interval; its top-1% and top-5% leads
  (10.1% vs 7.3%, 9.6% vs 7.6%) remain. `results.json` regenerated.
- `scripts/benchmark_models_vs_baselines.py`: the KDD Cup 1998 upgrade row
  (`bench_kdd_upgrade`) split its multi-year snapshot table at random by
  row, with `fiscal_year` as a feature. The file's gift log effectively ends
  in March 1996, so the FY1996 snapshot has almost no upgrades (3 in
  75,272), and the model scored AUC 0.902 mostly by learning which year a
  row came from; the same donor could also sit in train and test. It now
  walks forward (train FY1994, test FY1995, `fiscal_year` dropped as a
  feature). On the corrected split the upgrade model does not beat the best
  rule on this file: top-10% hit rate 11.0% vs 12.2% for "largest single
  gift in band", AUC 0.750 vs 0.750. The Results pages never showed a KDD98
  upgrade number, so no page changes.
- `scripts/make_results_pages.py` now records the git SHA and the installed
  `scikit-learn`/`numpy`/`pandas` versions in `docs/assets/results/results.json`
  (an `_env` key), so a number that doesn't reproduce can be traced to the
  environment it was generated under. The "Who to mail" page cited net
  revenue $4,382 (18,588 pieces mailed); regenerating under the recorded
  environment (scikit-learn 1.8.0, numpy 2.4.2, pandas 2.3.3) gives $4,240
  (18,748 pieces), and the $1K-upgrade page's worked example moves from 21 to
  19 upgrades in the model's top decile. Both pages now cite the regenerated
  numbers. `scripts/benchmark_models_vs_baselines.py --out dir/name` no
  longer crashes with `FileNotFoundError` when `dir/` does not already exist.
- `npsp_opportunities_to_features` summed every Opportunity that was not
  `Pledged`, so `Closed Lost` and open pipeline stages such as
  `Prospecting` counted as gifts. It now keeps only closed/won stages
  (`Closed Won`, `Awarded`, `Posted`) through `DEFAULT_INCLUDED_STAGES` and
  the caller-extendable `include_stages` parameter. `DEFAULT_EXCLUDED_STAGES`
  and `exclude_stages` are removed; neither was in a release.
- `philanthropy train`, `philanthropy score` and `philanthropy validate` fit
  and score on the named feature DataFrame instead of a bare array, so
  `score` and `validate` no longer print an sklearn "X does not have valid
  feature names" warning on every run. Saved model bundles are unaffected.
- `MajorGiftClassifier(class_weight=...)`: a new parameter for rebalancing
  rare major-gift labels (e.g. a 2-3% base rate), where `predict()` would
  otherwise favour the majority class almost exclusively. Applied as
  `sample_weight` during fitting, since handing the weight to the underlying
  `HistGradientBoostingClassifier` directly gets washed out (and can even
  invert the decision boundary) once `CalibratedClassifierCV` recalibrates
  probabilities from cross-validated folds.

### Fixed
- `philanthropy.ingest.map_columns` now raises a `ValueError` naming the
  colliding source columns when a mapping sends two different source columns
  to the same target name, instead of silently producing a duplicate-named
  output column that could still pass a `required=` check.
- `philanthropy.ingest._civicrm._to_amount` (shared by the CiviCRM, Raiser's
  Edge and NPSP readers) now treats an accounting-style parenthesised amount
  like `"($50.00)"` as negative instead of dropping the sign.
- `MajorGiftClassifier.predict_affinity_score` raised `IndexError` after a
  fit on single-class labels, because it indexed `predict_proba(X)[:, 1]`
  unconditionally. It now mirrors the single-class guard already used by
  `DonorPropensityModel.decision_function`.
- `DonorPropensityModel`'s docstrings described its `predict_proba` output as
  "calibrated" / "well-calibrated". It wraps a bare `RandomForestClassifier`
  with no calibration step and is measurably over-confident; the wording now
  matches the accurate note already on `predict_affinity_score`.

### Fixed
- `score_leadership_prospects` no longer crashes on a single-class historical
  target (no donor ever upgraded, or every one did) or on a training set too
  small for its internal 5-fold calibrated classifier; both now raise a
  clear `ValueError` instead of an opaque one from deep inside
  `CalibratedClassifierCV`.
- `score_leadership_prospects` dropped `fiscal_year` from the model's own
  feature columns (it isn't donor-specific, and the scored row's year always
  sits outside the training range); it's still returned as an output column.
- `score_leadership_prospects`'s validation report no longer uses a fixed
  top-10 count, which read as a 1.0 lift on a large real validation year
  whose true top-1% lift was 2.24x. `top_n` is now a parameter (default
  ~10% of the held-out fold), and the report adds `deciles`, `roc_auc`,
  `average_precision`, and two named baselines ("gave >= X last FY" and
  "top N by FY total") with their own rates and lifts.

### Fixed
- `npsp_opportunities_to_features` now resolves the donor key explicitly
  when an export carries both `AccountId`/`Account Name` and `Primary
  Contact`: the Account always wins, per NPSP's default Household Account
  model. Previously the two columns collapsed onto the same `contact_id`
  name and whichever happened to come first in the export's column order
  silently won.
- The docs homepage's quickstart example cited a stale held-out ROC-AUC
  (0.932) and major-donor count (347); the current code gives 0.841 and
  183. `docs/index.md`'s numbers, chart and table are regenerated to match.
- `docs/explanation/benchmarks.md`'s per-model accuracy table had one row
  (`DonorPropensityModel`) already regenerated under scikit-learn 1.8.0
  while the other three still carried their scikit-learn 1.7.2 numbers, and
  its footnote still cited 1.7.2. The whole table is now regenerated
  consistently to match the committed golden file
  (`docs/explanation/benchmark_results.txt`), and the footnote cites 1.8.0.

### Documentation
- README and the docs homepage now cover the Raiser's Edge and NPSP gift
  bridges, `read_gifts` as the one-call entry point over all three CRM
  presets, the `map_columns` / `activities_to_features` multi-file no-code
  upload path, and the CLI's `--activity`/`--as-of` and `train --task
  upgrade` flags for the leadership-upgrade model, none of which had been
  mentioned outside the API reference and the how-to guide.
- `scripts/make_results_pages.py`'s charts now state the finding in the
  title instead of only describing the axes (e.g. "Ranking by past giving
  beats the model: 79 of 100 vs 77 of 100 in the top 10%"), carry a
  direct end label on every bar so the value reads without an axis, show a
  5-seed range or a bootstrap interval as an error bar where one exists in
  `results.json`, draw "picking at random" as a dashed reference line (or a
  legend entry) instead of a third bar, format money results with a `$`
  and a thousands separator, and put the legend in a fixed header band so
  it never sits over a bar. The KDD98 lapse chart, where model, rule and
  random were all within two points of each other, is now a zoomed dot
  plot instead of three bars that looked identical. No result numbers
  changed; `docs/assets/results/results.json`'s values are byte-identical
  to before. `tests/test_results_docs_images.py` checks every Results page
  image has non-empty alt text and resolves to a real file.
- Results pages now lead with the finding, not the chart type, and losses
  get the same prominence as wins. Response leads with its two real-file
  wins (KDD98 held-out split and `cup98val`, both newly charted as
  `response_kdd98.png` / `response_cup98val.png`) instead of the synthetic
  loss it used to open with; the synthetic tab is now labelled "checks the
  code runs, not that the model works". Upgrade's verdict is now honest
  about the one real test available: it loses on KDD98 at a threshold
  rescaled to $50 (`bench_kdd_upgrade`, newly wired into
  `make_results_pages.py` as `upgrade_kdd98.png`), so the page says "wins
  on sample data, loses on the one real file, depends on your data"
  instead of a bare "Beats the simple rule". The decile chart is now a
  5-seed average with a range per bar (`upgrade_decile_average`) instead
  of one seed's noise, which is what actually produces the clean top-decile
  step the old caption claimed but the old single-seed chart did not show;
  the worked example below it now runs on the first of the same 5 seeds
  instead of an unrelated fixed seed, with one sentence explaining why its
  number still differs from the 5-seed average (a different feature
  pipeline, not a discrepancy). Lapse gains a retention read: a new
  `bench_kdd_lapse_retention` benchmark (same fit, ranked bottom-decile
  instead of top-decile, label flipped to "retained") shows the 10% of
  donors the model is least confident will lapse are a slightly better
  retention list than the same rule inverted, on a file where the top-decile
  lapse comparison is close to meaningless (95% base rate). Index verdicts
  updated to match.
- Who-to-mail's two-bar chart is now a profit curve: net revenue against how
  many donors are mailed, ranked most to least likely to respond, with the
  model's own "mail if expected gift beats the cost" stopping point and the
  mail-everyone endpoint both marked on the same line (new
  `kdd_mail_profit_curve`/`kdd_mail_profit_curve_val` in
  `benchmark_models_vs_baselines.py`, reusing the same fits and test splits
  `bench_kdd_cost_aware`/`bench_kdd_cost_aware_val` already used, so the
  numbers don't move). The page now leads with the persuasive sentence
  ("we skipped 9,920 of 28,624 letters and still raised $1,393 more")
  instead of burying it in paragraph two. Planned giving's stand-in chart
  (giving-response scored as if it were bequest intent) is dropped; the page
  now says plainly that no public bequest-intent dataset exists, what
  `PlannedGivingIntentScorer` would need to be tested for real, and how to
  run that test on your own file.
- New "scoreboard" figure (`_scoreboard_chart` in `make_results_pages.py`,
  `docs/assets/results/scoreboard.png`): one row per question, one dot per
  dataset, placed left/centre/right for loses/about the same/beats the
  simple rule, so a reader sees every model's win-or-loss verdict in one
  image instead of reading six pages. It leads the home page, the Results
  index, and the README, replacing the old wealth-screen-flavored donor
  table on the home page (invented names and dollar capacities) with donor
  IDs and a plain "gave again?" column. The home page's hero now asks the
  reader's actual question ("find out whether a model beats the rule your
  shop already uses, before you pay for one") instead of leading with
  `check_estimator` and Tier 1/2, which move to a new "For developers"
  section further down the page alongside the existing ROC-AUC code demo.
  New `docs/start_here.md`, a non-technical landing page ahead of Tutorials
  in the nav, walks through the same wins-and-losses story and links each
  finding to the page with the number behind it. Results moved to the
  second nav item, right after Home.
- New how-to guide `docs/how-to/get_your_crm_export_in.md`: one tab per
  `read_gifts` source (Raiser's Edge, Salesforce NPSP, Bloomerang,
  DonorPerfect, CiviCRM), each with the exact column mapping to
  `contact_id`/`receive_date`/`total_amount`, which column excludes pledges
  or failed contributions from the total, and a runnable example using each
  reader module's own already-verified sample rows.
  `tests/test_doc_examples.py`'s fenced-code extraction now dedents each
  block before executing it, since a code fence nested under a
  `pymdownx.tabbed` "===" tab is indented same as the rest of that tab's
  content; this doc is the first one to put a runnable example inside a
  tab.
### Fixed
- `activities_to_features`: `<type>_days_since_last` is now `NaN`, not 0, for
  a donor with no activity of that type at all; 0 read as "did it today"
  instead of "never". Counts and distinct still fill 0 for that case.
- `activities_to_features` raised `TypeError` when `as_of` was tz-aware (e.g.
  `pd.Timestamp("2024-12-31", tz="UTC")`); activity dates and `as_of` are now
  both normalised to naive UTC before comparison.
- `activities_to_features` stringified a float `contact_id` column (what
  `pd.read_csv` produces once any id cell is blank) as `"123.0"`, which then
  failed to join against the same donor's `"123"` from a column that never
  had a blank. Integral floats are now normalised to their bare digits first.

### Fixed
- `RFMTransformer` counted a gift with a NaN `gift_amount` toward `frequency`
  while silently dropping it from `monetary`, so the two columns described
  different sets of gifts. **Behaviour change:** a gift with no amount is now
  excluded from `frequency` as well as from `monetary`, with a `UserWarning`
  naming how many rows were dropped, so a donor's gift count can drop if some
  of their gifts have no recorded amount.
- `RFMTransformer.fit`/`transform` no longer pretend to accept a bare numpy
  array: the documented `x0..xn` column-naming path was dead code, since
  `_validate_input` always required `donor_id`/`gift_date`/`gift_amount` by
  name and so always raised on array input anyway. A numpy array now gets a
  clear `TypeError` up front instead of failing deeper in `_cut`/
  `_validate_input`. The docstring also now says outputs are raw R/F/M
  values, not scores or frozen bins.
- `LapsePredictor.classes_` is now built with `sklearn.utils.multiclass.
  unique_labels`, matching the other classifiers, instead of a bare
  `np.unique(y)`.
- `FinancialForecastModel`'s module docstring now names its actual backend
  (`LinearRegression` + `MLPRegressor` on the residuals + a hand-rolled
  AR(p) roll-forward) up front, alongside the "Hybrid LSTM-ARIMA" title,
  instead of only in the class docstring further down.

## [0.8.0] - 2026-09-24

The first release with a Raiser's Edge on-ramp and `as_of` scoring cutoffs on the
gift and encounter roll-ups. Everything under Breaking shipped in 0.7.0 and 0.7.1
emitting a `DeprecationWarning` naming this version.

### Breaking
- `FiscalYearGroupedSplitter(drop_repeat_donors=...)` now
  defaults to `True`, as the `DeprecationWarning` in 0.7.0 and 0.7.1 said it
  would. Each test fold drops donors already seen in its training rows, which
  is the safe default for a static per-donor label. It needs `groups` as
  `(n_samples, 2)` (fiscal year, donor id); code that passes fiscal years alone
  now raises a `ValueError` that names both fixes. Pass
  `drop_repeat_donors=False` for a time-varying target to keep the 0.7.x
  behaviour. The two leakage experiment scripts now pass it explicitly.
- Removed `philanthropy.utils.make_donor_dataset`, deprecated since
  0.7.0. Import it from `philanthropy.datasets`.
- Removed `WealthScreeningImputerKNN(group_col_idx=...)` and its
  `group_imputers_` attribute, deprecated since 0.7.0. Per-group and global KNN
  fits were measured bit-identical, so the parameter bought nothing; passing it
  is now a `TypeError`. `tests/test_knn_group_stratification.py` goes with it.

### Added
- `examples/notebooks/04_kdd98_end_to_end.ipynb` and the tutorial page
  "End to End on Real Donor Data": the whole library path on the 95,412 real
  donors of KDD Cup 1998, from a wide export to a cleaned gift log, a Raiser's
  Edge export with a pledge row, as-of RFM features, a lapse-model leakage
  backtest, a response model with permutation importance, a gift-size model
  with a calibrated interval and ask ladder, the mailing decision against the
  $0.68 piece cost, a disparity check, and a saved model bundle. It runs in CI
  with the other notebooks, and the downloaded archive is cached there.
- Added test coverage in `tests/test_encounter_timezone.py` guarding that
  `EncounterRecencyTransformer` raises a `KeyError` naming the invalid timezone
  and does not emit the misleading "Already tz-aware" error. Closes #204.
- Tests for `MovesManagementClassifier` now fit a named DataFrame and an
  array so `feature_names_in_` is both recorded and absent on the paths
  sklearn specifies. Closes #200.
- `philanthropy.ingest.raisers_edge_gifts_to_features` and
  `read_raisers_edge_gifts`: an on-ramp from a Blackbaud Raiser's Edge gift
  export to the donor-level feature table, alongside the existing CiviCRM and
  UniSchema bridges. Headers are normalised from any of the three spellings the
  product uses (desktop Export labels, the RE7 database columns, the RE NXT SKY
  API field names) onto the canonical `contact_id` / `receive_date` /
  `total_amount`, then the roll-up is delegated to the CiviCRM aggregator.
  The domain knowledge it adds is the commitment-versus-payment filter: in
  Raiser's Edge a pledge and the payments made against it are separate gift
  records, and a recurring gift row is a template rather than money received,
  so summing the amount column counts every committed dollar twice. The
  commitment rows and the ledger corrections are dropped by default, matched
  case-, space- and punctuation-insensitively so one spelling covers both the
  desktop and NXT vocabularies, with the desktop's abbreviated matching-gift
  types (`MG Pledge`, `MG Write Off`) named separately because they are not
  the long forms with the spaces taken out. The excluded set is the documented
  `exclude_gift_types` parameter, defaulting to the new
  `DEFAULT_EXCLUDED_GIFT_TYPES`, because Raiser's Edge exports are
  user-configured. An export with no gift-type column warns rather than
  silently double-counting. Closes #213.
- `philanthropy features --source {raisers_edge,civicrm} --data gifts.csv --out
  features.csv`, a fourth CLI subcommand. This is what makes the advertised
  no-Python path true end to end: previously `train` required `--features`
  columns such as `total_gift_amount` that nothing in the CLI could build, so
  "CSV in, scored CSV out" only held for someone who had already written the
  Python that produced them. The subcommand's `--help` lists the columns it
  emits, and the output runs through the same formula-injection neutralisation
  as `score` because a feature table echoes donor names and emails. It does not
  produce a label: `train --target` still needs a column the analyst defines.
- `EncounterRecencyTransformer` gains an `as_of` parameter, the last dated
  transformer without one. Encounters dated after it are blanked to `NaT`
  before the features are computed, so a clinical encounter that had not
  happened yet on the scoring date reads as a missing encounter instead of
  producing a negative `days_since_last_encounter`. It blanks rather than
  drops because the transformer emits one row per input row, and dropping
  would break it inside a `Pipeline`. Left at the default `None` nothing is
  blanked, but a post-reference-date encounter now warns instead of passing
  silently. Closes #210.
- `RFMTransformer` gains an `as_of` parameter. Gifts dated after it are dropped
  before the roll-up, so `frequency` and `monetary` describe only what had
  happened by the scoring date. The gift-side roll-up was the one feature
  builder that did not enforce the cutoff the clinical-encounter builders
  already did: with `reference_date` set and a gift table running past it,
  `monetary` silently summed gifts from after the date being scored and
  `recency` went negative. Left at the default `None` the behaviour is
  unchanged, but it now warns instead of aggregating the future silently.
  Closes #208.
- Added complete NumPy-formatted docstrings to `RFMTransformer.fit` and 
  `RFMTransformer.transform` methods, including detailed `Parameters`, 
  `Returns`, and `Raises` sections that now properly render in the 
  mkdocstrings-generated API reference.

- Added regression coverage for `PlannedGivingSignalTransformer` when transforming a NumPy array after fitting on a DataFrame.

- `tests/test_public_api_contract.py` gains two contracts over every public
  transformer, covering the two `get_feature_names_out` call shapes the suite
  never exercised. It previously only ever called `get_feature_names_out()` with
  no argument on a DataFrame-fitted transformer.
  `test_feature_names_out_accepts_input_features` passes the real column names
  through, which is the call `ColumnTransformer` and `Pipeline.get_feature_names_out`
  actually make, and `test_feature_names_out_width_after_array_fit` fits on an
  unnamed array. The second one is what caught the `CRMCleaner` defect above.
  Transformers that genuinely cannot fit on an unnamed array are listed in a new
  `_EXEMPT_ARRAY_FIT` table with a written reason each: `EncounterTransformer`
  merges on a named `donor_id`, and `MatchingGiftFeaturizer` rejects a
  non-DataFrame outright. That table is kept separate from `_EXEMPT` on purpose,
  because folding these two into `_EXEMPT` would also have dropped them from the
  width and `input_features` checks they do pass. Three hygiene tests police it:
  no stale entry, no name in both tables, and a reason on every entry in both.
- Notebooks 02 and 03 install the published wheel instead of a `git+main`
  snapshot. Both imported `datasets.make_donor_panel`, which 0.7.0 did not
  ship, and fell back to a `try/except ImportError` that pip-installed from
  `git+...@main`. That fallback could not work in-process: the failed
  `from philanthropy.datasets import make_donor_panel` leaves the stale module
  cached in `sys.modules`, so the re-import after a *successful* install raises
  the same `ImportError`. It only worked where philanthropy was absent
  entirely, which is fresh Colab, so anyone who followed the README's
  `pip install philanthropy` and then opened a notebook locally got a hard
  failure, and the "zero install, try it now" Colab badge ran an unreleased
  snapshot rather than the archived release `paper.md` points at. Now a plain
  `pip install -q "philanthropy[viz]>=0.7.1"`, which 0.7.1 satisfies from PyPI.
  Notebook 01 keeps its `try/except` because a bare `import philanthropy`
  succeeds against any release, so its guard never misfires.

### Changed
- The docs homepage hero no longer uses an all-caps eyebrow label or a
  gradient-clipped headline; it's now a two-column layout with the headline
  beside a real ranked-donor ledger table showing what
  `predict_affinity_score` returns. The "Key features" and "Explore the
  docs" sections moved from identical shadowed card grids to hairline
  ledger rows, with estimator class names right-aligned in mono for the
  feature list. The homepage also hides its table of contents so the hero
  gets the same content width as every other page.
- `AGENTS.md` brought back in line with the repository. The layout now lists
  `ingest/`, `inspection/` and `cli.py`; the dependency section names joblib
  as a runtime dependency and matplotlib/seaborn as the optional `viz` extra
  instead of claiming five runtime packages; the new-class workflow no longer
  hardcodes the 92% floor (which its own gate section forbids) or describes
  `make ci` as something it isn't, and says why the order matters. References
  to closed issues #21, #22 and #82 are gone, the merge section states the
  current single-maintainer rule without the history, `--no-verify` is banned
  once instead of three times, and the CONTRIBUTORS.md step now names the PR's
  human author and matches the PR template's opt-out.
- Seven high-intent documentation pages now carry a per-page `description` in
  YAML front matter, so a search result or social card shows that page's own
  summary rather than the site-wide `site_description`. The first-model
  tutorial is retitled "Building Your First Donor Propensity Model in Python";
  the previous "Building Your First Model" named neither the domain nor the
  language, and MkDocs derives the nav label from the first heading, so the
  entry was equally opaque there. Every description is quoted: an unquoted YAML
  scalar containing a colon and a space parses as a nested mapping, which
  silently drops the `<meta name="description">` tag and renders the front
  matter as visible page text instead.
- `paper.md`'s Research impact statement now cites the leakage preprint,
  archived on Zenodo as DOI `10.5281/zenodo.22665386`, alongside the real-data
  replication archive it already cited. `paper.bib` gains the matching
  `leakagepreprint2026` entry. The same statement's external-contribution
  count was stale and understated: it now reads thirty-five merged pull
  requests from eleven contributors external to the project, measured from the
  merged-PR list rather than recalled. Part of #127.
- The leakage tutorial's opening example is now a real one, contributed and
  attributed with permission: Marianne Pelletier's new-donor model built on a
  lifetime-giving-greater-than-zero variable, where the feature was the
  outcome. It replaces the synthetic `total_lifetime_giving` sketch and gains a
  runnable `as_of` walkthrough of the same failure. Closes #209.
- `CONTRIBUTING.md` and `docs/how-to/develop_and_test.md` now document the
  Windows-specific gaps in the local test/CI gate: `make ci`/`make riskcov`
  require `make`, which is not available in PowerShell by default, and
  `sh scripts/install_hooks.sh` requires Git Bash. Testing confirmed the
  installed pre-push hook does not fire when pushing from plain PowerShell.
  Closes #195.

### Removed
- Deleted `philanthropy/preprocessing/_solicitation_window.py`, a dead module
  nothing imported. The deprecated `SolicitationWindowTransformer` alias it held
  was already served by the subpackage's PEP 562 module-level `__getattr__`, so
  importing it still warns and still resolves to
  `DischargeToSolicitationWindowTransformer`. Closes #153.

### Fixed
- `ensure_local_path` now accepts absolute Windows drive-letter paths such as
  `C:\\data\\gifts.csv` without weakening rejection of network URLs. Closes #217.
- `EncounterRecencyTransformer` no longer catches timezone conversion errors
  and retries with `tz_localize` after parsing with `utc=True`. The retry was
  unreachable for valid timezone inputs and replaced the useful
  `UnknownTimeZoneError` for invalid timezone names with a misleading
  "Already tz-aware" error. Closes #201.
- `CRMCleaner.get_feature_names_out` raised `AttributeError: 'CRMCleaner' object
  has no attribute 'feature_names_in_'` when the transformer had been fitted on
  an unnamed array. `check_is_fitted` passed, because `n_features_in_` was set,
  and the next line then read an attribute that scikit-learn only assigns when
  the input carried column names. It now falls back to `x0`, `x1`, ... for an
  array fit, matching what `WealthScreeningImputer` and `WealthScreeningImputerKNN`
  already did. Closes #157.
- `MovesManagementClassifier` rejected NaN even though its
  `HistGradientBoostingClassifier` backend handles missing values natively;
  `fit`/`predict`/`predict_proba`/`action_priority` now pass NaN through.
  Fitting on a single-class `y` used to silently produce a classifier whose
  `predict_proba` returned only 1 column; it now raises a clear `ValueError`
  at fit time. Also removed a dead, already-redundant `feature_names_in_`
  assignment (`validate_data` sets it) and documented that
  `action_priority`'s confidence is an uncalibrated max probability.
- `PlannedGivingIntentScorer.fit` raised scikit-learn's raw "Requesting
  2-fold cross-validation..." error when a class had fewer than 2 examples,
  because calibration uses a fixed `cv=2`. It now raises a clear `ValueError`
  before calibration runs. Also removed an unreachable branch in
  `predict_intent_score` (`predict_proba` always returns 2 columns once
  `fit` requires at least 2 classes) and documented that NaN features are
  rejected.

## [0.7.1] - 2026-09-08

### Added
- `scripts/render_leakage_chart.py`, the figure companion to the two leakage
  experiments. `leakage_experiment.py` and `real_data_leakage_experiment.py`
  each print five-seed means; this script re-runs the identical
  cross-validation and keeps the per-fold scores, so the shape of the
  walk-forward backtest is visible rather than one number per condition. It
  writes a two-panel figure (synthetic and KDD Cup 1998, as-of against
  whole-history features, seed spread shaded) plus the numbers as JSON, and
  reproduces the published figures: 0.625 against 0.750 on the synthetic panel,
  0.482 against 0.858 on the real one. `--cached` re-plots from a saved score
  file so styling changes do not refit the real panel, and `--out` chooses the
  output directory. The rendered files are gitignored rather than committed.
- `README.md` gets a `### Prior art` section under Research, crediting the R
  repositories this package is downstream of: `michaelpawlus/pg_donors` (2015),
  `michaelpawlus/fundraising_analytics` (2016), and `crazybilly/fundRaising`
  (2021). `PlannedGivingIntentScorer` is named as the descendant of `pg_donors`,
  and `RFMTransformer`, `DonorPropensityModel`, `MovesManagementClassifier`,
  `FiscalYearTransformer`, and `LapsePredictor` are matched to the R scripts and
  functions that did the same job first. The omission read as a state-of-the-field
  gap: `paper.md` compares against Python libraries only, and nothing anywhere in
  the project acknowledged the R prior art.
- `CONTRIBUTING.md` gets a "Claiming an issue" rule: comment on an issue before
  opening a PR, and wait for a maintainer to assign it. Issues #175 and #176
  were two people independently fixing the same four-line bug the same
  evening; that was a process failure on this project's side, not a mistake
  by either contributor. The issue-draft template now states the same rule
  inline, so every newly filed `good first issue` carries the reminder in its
  own body rather than depending on a reader following a link.

- Tests pinning the two untested branches of `WealthPercentileTransformer`:
  the all-missing column path (returns NaN ranks, keeps a stable output
  width, raises no warning) and the partially-missing column path (NaN
  input rows get NaN rank, observed rows get numeric rank). Closes #169.
- **`datasets.make_donor_panel`** (Tier 2, Beta): a seeded multi-year donor
  panel returning gift-level rows rather than one aggregated row per donor.
  `generate_synthetic_donor_data` cannot demonstrate `RFMTransformer` (needs a
  gift log), `FiscalYearGroupedSplitter` (needs repeated donor-years), an
  `as_of` cutoff (needs something to cut off), or the grateful-patient
  transformers (need encounters), so the only generator that could show the
  library's central ideas lived privately inside
  `scripts/leakage_experiment.py`. This promotes it.
  - Returns `{"gifts", "donors"}`, plus `"encounters"` when
    `include_encounters=True`. Column names match what the transformers
    already require, so nothing has to be renamed on the way in.
  - Fiscal years run 1 July to 30 June, labelled by the year they end in. At
    most one gift per donor-year, so "recent" is well defined.
  - **No label column, deliberately.** A label is a claim about a point in
    time, and shipping one pre-computed hands every user the exact mistake this
    package exists to prevent. The docstring shows the one-line derivation.
  - `wealth_estimate` is ~30% missing by design, because a wealth screen that
    came back for every record is not a wealth screen anyone has received.
  - `scripts/leakage_experiment.py` now imports it instead of defining a
    private copy, so the published experiment and the tutorials run on the same
    generator. The experiment's numbers are unchanged, and not merely to three
    decimals: the aggregated frames are asserted byte-identical to the ones the
    private generator produced, on all five published seeds. Gift amounts are
    deliberately **not** rounded to cents for that reason; rounding moved the
    reported min-max ranges by 0.001 AUC.
- **Three notebooks under `examples/notebooks/`**, each with a Colab badge,
  executed end to end in CI on every push (`pytest --nbmake examples/notebooks`,
  one leg of the `lint` job): `01_quickstart_propensity.ipynb` (the README
  quickstart plus a call list, a distribution plot, and permutation
  importance), `02_temporal_leakage.ipynb` (builds `make_donor_panel`'s
  features as-of and over the whole export and measures the inflation, an
  optional cell behind `PHILANTHROPY_FETCH_KDD98` reproduces the real-data
  number), and `03_grateful_patient_pipeline.ipynb` (encounters, an `as_of`
  cutoff, service-line weighting, the solicitation window, routed through a
  `ColumnTransformer` rather than a serial `Pipeline`, with an assertion that
  the pipeline is not degenerate). `nbmake>=1.5` added to the `dev` extra as a
  dev-only dependency; the runtime dependency rule (scikit-learn, pandas,
  numpy, matplotlib, seaborn, nothing else) is untouched.
  `examples/quickstart.ipynb` becomes a one-cell redirect to notebook 01 for
  one release, so the previous Colab badge and any existing links keep
  working; `tests/test_examples.py`'s docstring now says notebooks are covered
  by `nbmake`, not by it.

- `credit-guard` CI job: pull requests touching `philanthropy/` must also
  update this changelog, and the author must be credited in
  CONTRIBUTORS.md. Implemented as `scripts/check_credit.sh`, wired into
  `ci.yml` on `pull_request` events only; failures surface as inline
  `::error::` annotations on the Files tab. Closes #113.
- Regression coverage ensuring `MovesManagementClassifier.fit` preserves
  DataFrame column names in `feature_names_in_`. Closes #54.
- `.github/workflows/pypi-smoke.yml`: a weekly (Mondays 12:00 UTC) and
  manually dispatchable job that installs the **published wheel** from PyPI on
  Linux, macOS and Windows, then runs `examples/quickstart.py` and
  `philanthropy --help`. Every other job tests the working tree; this one tests
  what `pip install philanthropy` actually serves, which is the only thing a new
  user or a reviewer runs. The repository is checked out into a subdirectory and
  the job asserts `philanthropy.__file__` resolves inside `site-packages`, so a
  `philanthropy/` directory in the working directory cannot silently shadow the
  wheel and turn this into a second working-tree test. Windows is included
  because the main matrix is Linux plus macOS.
- **`models.GiftIntervalCalibrator`**: distribution-free intervals on a dollar
  amount. Wraps an already-fitted regressor (`AskAmountRecommender`,
  `ShareOfWalletRegressor`, or any `predict`-per-row estimator) and calibrates on
  held-out rows via split conformal prediction. Until now nothing in the package
  returned an interval on a gift amount; every dollar-valued estimator returned
  a point.
  - Refuses below the certification floor. One order statistic needs
    `n >= 1/alpha - 1` calibration rows, 19 at the 95 % level, and `fit` raises
    rather than returning an interval it cannot certify. The floor is computed in
    `fractions.Fraction`, because it is a ceiling and `int(1 / alpha - 1)`
    truncates: at `alpha = 0.07` that reports 13 where the floor is 14. There is
    no parameter that switches the check off.
  - Reports the **attained** level, `r / (n + 1)`, on the returned
    `GiftInterval` and as `attained_level_`. A request for 0.95 resolves to
    0.9677 at 30 calibration rows and 0.9524 at 20; the requested level is kept
    separately as `requested_level_`.
  - Three one-rank conformity scores via `score=`: `"absolute"`,
    `"difficulty"` (residual over a difficulty estimate) and `"log"` (residual
    on `log1p` dollars, inverted). Equal-tailed two-rank intervals are
    deliberately not offered: two order statistics at `alpha / 2` more than double the
    floor to 39 rows and buy nothing the one-rank forms do not.
  - Intersects the interval with `[lower_bound, inf)`, default `0.0`. A gift
    cannot be negative, so coverage is bit-identical and width strictly falls.
    Calibration targets below the bound raise, since they are evidence the bound
    is wrong.
  - Optional `groups=` calibrates within a segment. A pooled calibration set is
    dominated by whichever segment supplies most of the rows and under-covers the
    others however much marginal data is added; a group below the per-group floor
    is refused by name rather than quietly pooled with a segment at another
    capacity level.
- **`metrics.interval_score` and `metrics.interval_report`**: the interval score
  `(u - l) + (2/alpha)(l - y)+ + (2/alpha)(y - u)+`, which is proper for a
  central interval, plus a report carrying coverage, the score as mean/median/
  trimmed mean (it is a heavy-tailed loss on gift amounts, and a ranking that
  flips between the three is a ranking of the tail), median width, and
  `width_ratio` = median width over median target. A valid interval can carry no
  information; the ratio is what separates the two.
- Test coverage for `PlannedGivingIntentScorer.predict_intent_score`: the
  single-class `predict_proba` fallback path that returns an all-zero score.
  (#56)

### Changed
- The `ShareOfWalletScorer` output rename describes itself as landing in 0.7.1
  rather than 0.8.0. The docstring, the `get_legacy_feature_names_out` summary
  line, its `DeprecationWarning` text, and the deprecations table in
  `docs/reference/index.md` all named 0.8.0, which was the version this work was
  staged for before the release was cut as a patch. The three future promises
  are untouched and still say 0.8.0: `WealthScreeningImputerKNN(group_col_idx=...)`,
  `philanthropy.utils.make_donor_dataset`, and the `FiscalYearGroupedSplitter`
  `drop_repeat_donors` default flip. Removal of the legacy names accessor stays
  at 0.9.0, which is more grace than the one-published-minor rule requires.
- Generative AI disclosure reordered in `paper.md`, `README.md`, and
  `philanthropy/__init__.py` to lead with human design authority and human
  review, then state the scope of the assistance within those constraints. The
  facts are unchanged: the scope remains package-wide, no numeric split is
  estimated, and the review gate is still what the disclosure rests on. Only the
  order and emphasis moved, so the disclosure stays consistent with `AGENTS.md`
  and the public commit history.
- `WealthPercentileTransformer.fit` now raises an actionable `ValueError` when
  an explicit `wealth_cols` list matches no training column; partial matches
  and automatic detection remain unchanged.
- README coverage badge now links to `pyproject.toml` and reads "≥92% floor"
  rather than a bare "≥92%". It was a static shields.io string with no tie to
  the enforced number, so it would have silently lied had `fail_under` ever
  moved. Not wiring a dynamic gist badge (`schneegans/dynamic-badges-action`):
  that needs a personal-access-token secret this change doesn't have standing
  to create, and Codecov/Coveralls are explicitly ruled out elsewhere in this
  project's standing rules.

- `ShareOfWalletScorer` output column 0 is renamed `sow_score` →
  `capacity_utilisation_ratio`. The formula was always capacity ÷ clipped
  modelled wealth: utilisation of estimated capacity, with no term for giving to
  *your* institution, so the old name claimed a share-of-wallet quantity the
  score cannot express (the class docstring has warned about exactly this since
  it shipped). Values, column order, and `capacity_tier` are unchanged; code
  reading column 0 positionally needs nothing. Code spelling the name gets one
  published minor of grace via `get_legacy_feature_names_out()`, which returns
  the old `["sow_score", "capacity_tier"]` under a `DeprecationWarning` and is
  removed in 0.9.0; the shim is registered in `tests/test_deprecations.py`.
  Closes #109.
- `predict_<thing>_interval` joins `_score` and `_forecast` as an accepted
  domain-method suffix in the public-API naming contract
  (`tests/test_public_api_contract.py`, `AGENTS.md`).
- `test_predict_methods_are_callable_with_x_alone_and_return_one_value_per_row`
  now skips non-estimator symbols in `models.__all__` instead of raising
  `KeyError` on the first one.
- `CRMCleaner` and `FiscalYearTransformer` now use a shared `_validate_X` helper,
  and the unreachable `np.iscomplexobj` guard is removed. Complex data inside
  object arrays bypasses the guard and is properly handled downstream. Closes #155.

### Deprecated
- `FiscalYearGroupedSplitter`'s default for `drop_repeat_donors` (currently `False`) is deprecated and will change to `True` in 0.8.0. Leaving it at its default now emits a `DeprecationWarning`. Pass `drop_repeat_donors=False` explicitly to silence the warning and retain current behavior. Closes #108, by @shubhrai23.

- `philanthropy.utils.make_donor_dataset` moves to
  [`philanthropy.datasets.make_donor_dataset`](philanthropy/datasets/) and the
  old location emits a `DeprecationWarning`; removed in 0.8.0. The gift-level
  generator now lives next to `generate_synthetic_donor_data`, which is the
  canonical datasets home. Closes #111.

### Fixed
- `GratefulPatientFeaturizer` now reports one fallback `general` service line
  per known donor when the encounter table omits the service-line column.
  Missing physician columns continue to report zero distinct physicians, and
  both optional-column paths now have regression coverage. Closes #151.

- `EncounterTransformer` now accepts parsed `datetime64` gift dates alongside
  date strings, and reports unparseable values with the configured gift-date
  column name instead of exposing NumPy's mixed-dtype promotion error. Closes
  #163.
- `CRMCleaner.transform` no longer silently corrupts complex amounts into
  wrong finite floats: cells holding actual `complex` values are masked to
  NaN with a `UserWarning` naming them, and a column where nothing parses
  (all-complex included) still raises `could not parse` per the documented
  contract. Closes #129.

## [1.0.0] - TBD

The API freeze. No code changes: 1.0.0 is a promise, not a feature.

### Changed
- `Development Status :: 5 - Production/Stable`.
- **Tier 1 is now semver-protected.** A breaking change to any Tier 1 symbol
  requires a major release, preceded by one full published minor emitting
  `DeprecationWarning`. Tier 2 may still break in a minor; Tier 3 carries no
  guarantee. The tiers are listed per-symbol in
  [docs/reference/index.md](docs/reference/index.md).

### Added
- `test_stability_tier_table_covers_every_public_symbol`: the tier table is now
  machine-checked against `__all__`, so a new public symbol cannot ship without
  a stated tier. At 1.0 that table is the contract; an out-of-date one is a
  broken promise, not a docs nit.

### Notes
The five 1.0 gates all hold at this commit: 0.7.0 published; the public-API
contract test green with no exemption added since 0.7.0; no `deprecated_alias`
anywhere in `philanthropy/`; `__version__ == importlib.metadata.version(...)`
and `py.typed` in the wheel; every `__all__` symbol carries a tier and no Tier 1
entry is mid-deprecation.

### Documentation
- New page **Real-Data Replication: KDD Cup 1998**
  (`docs/explanation/real_data_replication.md`), promoted out of a section of
  `benchmarks.md` and expanded: synthetic and real numbers side by side, the
  panel construction from the wide promotion history, the pre-registered
  prediction that was wrong by a factor of five, the download caveat, and the
  Zenodo replication DOI. `benchmarks.md` keeps the headline tables and links
  out, so the measured numbers still live in exactly one place. The README
  gains a "Validated on real donor data" section pointing at it.
- "Which estimator do I need?" table at the top of `docs/tutorials/index.md`,
  keyed by the question a fundraising shop actually asks rather than by module.
  Fifteen rows covering every Tier 1 and Tier 2 estimator plus the metrics, with
  the required data shape in the middle column, because that is usually the real
  work. Closes the gap where a reader had to infer the entry point from the
  feature tables.

### Typing
- **mypy ratchet finished.** `py.typed` ships in the wheel, so a user's type
  checker treats every unannotated function here as `Any`, which is worse than
  shipping no type information: it silently disables checking at the boundary
  instead of admitting there is nothing to check. `experimental`, `models` and
  `preprocessing`, the last three subpackages, are now fully annotated: `fit`
  returns a per-class `TypeVar` bound to the class rather than a string
  literal, so a subclass's `fit` no longer reports its parent's type;
  `__sklearn_tags__` returns sklearn's public `Tags` dataclass
  (`scikit-learn>=1.6`, the declared floor). With every subpackage covered,
  the `[[tool.mypy.overrides]]` block is gone and `disallow_untyped_defs =
  true` is a top-level `[tool.mypy]` setting, so a newly-added unannotated
  function anywhere in `philanthropy` fails CI. Closes #166.
- `ensure_local_path` is now generic in its argument (`TypeVar`) rather than
  declared `-> str`. It returns its input unchanged, and both call sites pass
  something that may be a `Path`, so the old annotation was a small lie; the
  docstring said "the unchanged path" and now the type says so too.
### Changed
- Issue templates converted from Markdown to **YAML issue forms**
  (`bug_report.yml`, `feature_request.yml`). The Markdown versions asked for a
  version and a reproducer and could be submitted without either, and GitHub's
  community profile reported `issue_template: false` because it counts only
  forms. The bug form requires the version, the environment line and a runnable
  reproducer, plus an explicit tick that the reproducer contains no real donor
  or patient data. The feature form states the two constraints that decide most
  requests (frozen dependency set, no network in the core) before the author
  starts writing. `config.yml` is unchanged.

## [0.7.0] - 2026-08-21

The removal release, plus everything else merged since 0.6.0. Every shim
under Breaking shipped in 0.6.0 emitting a `DeprecationWarning` for one full
published minor; everything under Added, Changed and Deprecated below is new
work that ships for the first time in this release.

### Breaking
- **Four deprecated method aliases removed.** Use the replacement in every case:

  | Removed | Use instead |
  |---|---|
  | `AskAmountRecommender.predict_ask_array` | `ask_ladder` |
  | `ShareOfWalletRegressor.predict_capacity_ratio` | `capacity_ratio` |
  | `MovesManagementClassifier.predict_action_priority` | `action_priority` |
  | `PlannedGivingIntentScorer.predict_bequest_intent_score` | `predict_intent_score` |

- **Three dead constructor parameters removed.** Passing any of them is now a
  `TypeError`: `LapsePredictor(lapse_window_years=...)` (the window is a
  property of how you labelled `y`), `PropensityScorer(estimator=...)` (the
  baseline is a constant 0.5), `FiscalYearGroupedSplitter(fiscal_year_start=...)`
  (`groups` already carries fiscal-year labels).
- **`donor_acquisition_cost`, `cost_per_dollar_raised` and `fundraising_roi` are
  keyword-only.** They do not share an argument order:
  `cost_per_dollar_raised` takes expense first, `fundraising_roi` takes raised
  first, so a positional call was silently accepted and returned a plausible
  wrong number. It is now a `TypeError`.
- **Four accidental second import paths moved behind underscores** so 1.0 does
  not freeze them: `metrics.scoring` → `metrics._scoring`,
  `preprocessing.transformers` → `preprocessing._transformers`,
  `models.propensity` → `models._propensity_baseline`, `utils.testing` →
  `utils._testing`. Every public symbol is unchanged and still exported from its
  subpackage; only a direct `from philanthropy.metrics.scoring import ...`
  breaks. Import from the subpackage instead.
- `philanthropy/utils/_deprecation.py` is gone. `tests/test_deprecations.py`,
  which existed solely to police the shims removed above, went with it, then
  came back later in this same release to police a new one; see Deprecated
  below.

### Changed
- `GratefulPatientFeaturizer`, `EncounterTransformer` and the `philanthropy`
  CLI now reject network-scheme paths (`https://`, `s3://`, `gs://`) with a
  `ValueError` before any file read. Previously the no-network guarantee held
  for the library's own logic but not for its documented public parameters,
  because `pandas` will follow a remote URI if handed one. Local paths,
  including `file://`, are unaffected. Closes #114.
- The no-network promise in `README.md`, `SECURITY.md` and the security review
  Q&A is now stated as two precise guarantees, "never transmits your data" and
  "downloads nothing", instead of the blanket "no network calls of any kind",
  and the second one is machine-checked. `tests/test_no_network.py` now parses
  every module in the package and fails the build if one imports a
  network-capable library without appearing on an explicit allowlist. The
  allowlist is empty, so the effective promise is unchanged and is now enforced
  across modules no test happens to import, rather than only on the paths the
  socket fixture walks.

### Added
- Question 1a in the security review Q&A documents the remote-path rejection,
  which is the behaviour a privacy officer asks about after reading question 1.
- `philanthropy.datasets.fetch_kdd98_donors`, an opt-in fetcher for the KDD Cup
  1998 direct-mail donor dataset, cached locally after first download. It is
  the one entry in the no-network allowlist added above, and it exists so the
  library can be validated against real donor data instead of only synthetic
  data. Part of #124.
- `scripts/real_data_leakage_experiment.py` replicates `leakage_experiment.py`
  on the real KDD Cup 1998 file instead of the synthetic panel. The predicted
  effect (recorded in the script before it was run) was smaller than the
  synthetic numbers; the measured effect is larger: whole-history feature
  construction inflates walk-forward ROC-AUC by +0.376 AUC (versus +0.126
  synthetic), and a random `StratifiedKFold` split overstates the true future
  by +0.107 AUC (versus 0.014-0.030 synthetic). Documented in
  `docs/explanation/benchmarks.md` and in `paper.md`'s Statement of need and
  Research impact statement. The script outputs and environment lock are
  archived on Zenodo (DOI [10.5281/zenodo.22050649](https://doi.org/10.5281/zenodo.22050649)).
  Closes #124.

### Deprecated
- `WealthScreeningImputerKNN(group_col_idx=...)` is **deprecated** and will be
  removed in 0.8.0. It still works and now emits a `DeprecationWarning`. There is
  no replacement because there is nothing to replace: measured across several
  synthetic two-group pools, and on five Python versions in CI, per-group and
  global KNN imputation produce bit-identical output (`50263.48615163204` both
  ways). A donor's nearest neighbours by feature distance almost always share
  their group already, and `KNNImputer` weights distance by column magnitude, so
  a 0/1 group flag barely registers. The parameter costs a per-group imputer,
  three fallback paths and a documented contract, and buys no measurable
  accuracy. Split the frame by group and fit one imputer per part if you need
  that behaviour. `tests/test_deprecations.py` is reintroduced per `RELEASING.md`,
  with the registry meta-test that fails when a shim ships untested; it walks the
  package AST for `warnings.warn(..., DeprecationWarning)` call sites, so a
  docstring merely mentioning the class is not miscounted. Closes #85.

### Added
- `paper.md` now carries the four JOSS sections it was missing: **State of the
  field**, **Software design**, **Research impact statement**, and **AI usage
  disclosure**. JOSS made all six sections required and moved the length window
  to 750-1750 words in January 2026; the paper was 635 words with two of six
  sections, which is a pre-review bounce on its own. It is now 1447 words. The
  AI usage disclosure restates the one already in `README.md` and in
  `philanthropy/__init__.py`, since JOSS requires it as a named section of the
  paper itself.
- `paper.bib` gains the prior art the paper had never cited: feature-engine,
  mlxtend, sktime, pymc-marketing, MAPIE, crepes, Fader-Hardie-Lee (BTYD),
  Zhang (2003) for the linear-plus-nonlinear forecast decomposition, and Bates
  et al. (2023) for the conformal p-value the code already attributes to it.
  JOSS accepts re-implementations "provided that they cite prior similar work",
  and the bibliography previously cited no comparable package.
- `RFMTransformer(include_tenure=True)` emits a fifth column, `tenure`: days
  from the donor's first gift to the frozen reference date. Recency, frequency
  and monetary alone cannot feed a buy-till-you-die model, which needs the
  observation window T as well. Defaults to False so the output shape does not
  move under existing callers.
- `ShareOfWalletScorer(major_tier_threshold=..., principal_tier_threshold=...)`.
  The 0.40 and 0.75 cut points were hardcoded in `transform` with no source and
  no way to match an institution's own tiering.
- `DischargeToSolicitationWindowTransformer(window_shape=...)`, with the legacy
  symmetric triangle available as `"triangle"` for reproducing older runs.
- `EncounterTransformer.fit` warns when `as_of=None` and the encounter table
  contains discharges later than the latest gift date in `X`, naming the row
  count. That is the one leakage path no cross-validation splitter can see: the
  encounter table is a constructor argument, so its rows are never part of any
  split.
- `GratefulPatientFeaturizer` warns when `drg_weight_col` is set. A DRG relative
  weight is diagnosis-derived, and diagnosis is not in the element list the HIPAA
  fundraising carve-out permits (45 CFR 164.514(f)).

### Changed
- **Behaviour change.** `DischargeToSolicitationWindowTransformer` now decays
  `window_position_score` from 1.0 at `min_days_post_discharge` to 0.0 at
  `max_days_post_discharge` instead of peaking at the window midpoint. The old
  symmetric triangle treated the ethical cooling-off floor as a propensity
  minimum: with the default 90-365 window, day 91 and day 364 both scored about
  0.007 while day 227 scored 1.0. Pass `window_shape="triangle"` to reproduce
  the previous numbers.
- **Behaviour change.** A missing days-since-discharge value now yields
  `window_position_score=NaN` rather than `0.0`, so "no discharge on record" is
  distinguishable from "discharged, but outside the window", which still scores a
  hard 0.0. `in_solicitation_window` is unchanged at 0.
- **Behaviour change.** `GratefulPatientFeaturizer(use_capacity_weights=...)`
  now defaults to `False`. The built-in service-line multipliers have no
  published source, and defaulting them on meant the headline
  `clinical_gravity_score` silently carried unsourced 2.7x to 3.2x weighting.
- `EncounterTransformer.dropped_cols_` now includes `gift_date_col`, which
  `transform` drops separately. `compliance_considerations.md` tells operators to
  inspect this attribute as their audit trail, and it was under-reporting what
  actually left.
- `philanthropy.preprocessing.SolicitationWindowTransformer` is deprecated and
  emits a `DeprecationWarning` on access via PEP 562 module `__getattr__`. It
  still resolves to `DischargeToSolicitationWindowTransformer` itself, so
  `isinstance` and `clone` are unaffected, and it is registered in
  `tests/test_deprecations.py` for removal in 1.0.0. Two public names for one
  transformer inflated the API surface without adding capability.
- `CITATION.cff` and `.zenodo.json` now match `paper.md` on title and author
  name, and both carry the ORCID. `CITATION.cff` records `version: 0.6.0`, the
  release the concept DOI actually resolves to, instead of the in-development
  `1.0.0` from `pyproject.toml`. A reviewer following the archive DOI was landing
  on a record that contradicted the paper byline.

### Fixed
- Four claims in `paper.md` that were falsifiable by running the code. The
  conformance claim named `UpliftTLearner` as "the one documented exception"
  against four entries in `_MANUALLY_COVERED`; the leakage claim said a
  PhilanthroPy pipeline "cannot leak test-period or future information" while
  `_encounters.py` documents that exact leak at the default `as_of=None`;
  fiscal-year boundaries were listed as a frozen fitted statistic although
  `FiscalYearTransformer` has no fitted state (its own test is named
  `test_fiscal_year_stateless`); and "compose directly inside
  `sklearn.pipeline.Pipeline`" did not exclude the row-reducing
  `RFMTransformer`. The Summary now describes the conformance registry as the
  mechanism it is: 20 configured instances, 1016 checks on scikit-learn 1.8.0,
  four documented exemptions, and a build-failing guard against a public
  estimator appearing in neither list.
- `conformal_pvalue`: thresholding at `alpha` bounds the expected **selection
  rate**, not the false-positive rate. The FPR reading needs a calibration set
  of nulls only, which is the construction in Bates et al. (2023) and not what
  "donors held out of training" gives you. The wrong statement was in the shipped
  module docstring and therefore in the rendered API docs, not only in the paper.
- The `check_estimator` claim was corrected in `paper.md` but still stood in
  three other places, in its strongest and most falsifiable form: `README.md`
  ("Every public estimator passes `check_estimator`", with the `UpliftTLearner`
  qualification trimmed off at some point), `docs/explanation/design_principles.md`
  ("the one exception"), and `docs/explanation/security_review_answers.md`, which
  is the page written to be forwarded to a privacy or procurement reviewer. All
  three now describe the battery, its four documented exemptions, and the
  build-failing guard, matching the paper.
- `EncounterRecencyTransformer` described itself as producing "HIPAA-safe"
  features in four places. Date-only input is not de-identified: Safe Harbor
  strips every date element more granular than a year, so encounter dates are
  themselves identifiers, permitted for fundraising only under the narrower
  164.514(f) carve-out. This contradicted the project's own compliance page.
- `security_review_answers.md` attributed `PII_PATTERNS` column-dropping to
  `CRMCleaner`, which has neither the attribute nor any dropping logic. It lives
  on `EncounterTransformer`. That page exists to be forwarded to a privacy
  officer, so the error cost more than a docs bug normally would. It now also
  states that `pii_patterns` replaces rather than extends the defaults.
- `ShareOfWalletScorer`'s docstring claimed a share of wallet. The formula has no
  term for giving to your institution anywhere in it, so it cannot express what
  fraction of a donor's philanthropy you receive; it is capacity over modelled
  wealth. `docs/index.md` repeated the wrong definition. The output name is kept
  for compatibility and flagged for renaming in the next major release. The
  fit-time 95th-percentile denominator clip, which inflates exactly the top tier,
  is now documented rather than silent.
- The README figure caption said the affinity scores "cleanly separate major from
  non-major donors", 21 lines below the text explaining that the distributions
  overlap. That was the in-sample overclaim commit c54a6b3 retracted, left behind
  in the caption.
- `AskAmountRecommender` and `ShareOfWalletRegressor` now say in their docstrings
  that they are the same `HistGradientBoostingRegressor` wrapper with different
  targets, and `PropensityScorer` that it is equivalent in effect to
  `DummyClassifier(strategy="uniform")`. Four classes exposed `proba * 100`
  under four names with nothing saying they were the same thing.

### Notes
- `paper.md` cites `scripts/leakage_experiment.py` and its measured result
  (whole-history feature aggregation inflates walk-forward ROC-AUC from 0.625 to
  0.750, +0.126, against 0.014 and 0.030 of splitter-choice error). That script
  arrives with #101, so #101 must land for the reference to resolve. The numbers
  above were reproduced locally against this branch before being written down.
- Still open, and not fixable in a pull request: JOSS requires demonstrated
  research impact, and no estimator here has ever been fitted on real donor
  data. `load_ciob_fundraising` carries no donor rows, amounts or labels. The
  paper's Research impact statement says so plainly rather than implying
  adoption that does not exist.
- `WealthScreeningImputerKNN.group_col_idx` now does what it always claimed.
  It was documented as stratifying KNN imputation per group "improving local
  accuracy", and was stored and never read. When set with `strategy="knn"`, a
  separate `KNNImputer` is now fitted per group, so a donor's missing wealth is
  filled from neighbours inside their own group instead of from the whole
  database. **The measured benefit is small and setup-dependent**, which is worth
  saying plainly given the old docstring promised "improving local accuracy":
  across several synthetic two-group pools the grouped and global fits often
  agree **exactly**, because a donor's nearest neighbours by feature distance
  usually share their group already, and `KNNImputer`'s distance is dominated by
  large-magnitude columns so a 0/1 group flag contributes little either way. CI
  demonstrated this on other numpy/sklearn versions, where the two fills came out
  bit-identical, so no test here asserts that grouping changes a value. The
  honest case for the parameter is explicit control, not a demonstrated accuracy
  gain, and issue #85's option B (deprecate it) remains defensible on that basis. Three fallbacks are frozen at fit time so nothing is learned at transform
  time: a group with fewer than `n_neighbors + 1` training rows gets no imputer of
  its own; a group value unseen at fit, or a row whose group label is missing,
  uses the global imputer; and a column entirely missing *within* a group also
  defers to the global imputer, because
  `KNNImputer(keep_empty_features=True)` fills such a column with a hard `0.0`
  rather than `NaN`, which for a wealth column reads as "no capacity" and would be
  a materially wrong number for every donor in that group. The global imputer is
  always fitted, so output is never `NaN` regardless of grouping. Ignored for the
  columnwise strategies, which have no notion of a neighbourhood. An out-of-range
  index raises, for `strategy="knn"` where the parameter has any effect.
  Closes #85.
- `FiscalYearGroupedSplitter(drop_repeat_donors=True)` for the static-per-donor
  label case. The splitter groups by fiscal year, correctly, but not by donor, so
  a donor with gifts in several fiscal years lands in both folds of a split. That
  is right for a time-varying target and is leakage for a static label such as
  `is_major_donor`, which is the label used throughout the README, the benchmarks
  page and `scripts/benchmark_models.py`. With the flag set, each test fold drops
  donors already present in its training rows; `groups` then takes shape
  `(n_samples, 2)` with the donor identifier in column 1. Training rows are never
  dropped. The cost is made visible rather than silent: `split` warns with the
  number of test rows removed, and notes that the remaining test donors are
  systematically newer to the file. A test fold emptied entirely raises with an
  actionable message rather than being skipped, which would have put `split` and
  `get_n_splits` back out of step. A row with a **missing** donor id is treated as
  already-seen and dropped: `np.isin` never matches `NaN` to `NaN`, so it would
  otherwise have been kept, and an unidentifiable donor cannot be shown to be
  absent from training. A string-typed `groups` (which is what
  `np.column_stack` produces from integer years and string donor ids) has its
  fiscal-year column coerced back to numeric, and a genuinely non-numeric year
  column now raises with an actionable message instead of a bare numpy
  `TypeError`. `__repr__` includes the flag, so two splitters that split
  differently no longer print identically. Defaults to `False`, so nothing
  changes until you opt in. Part of #87; the docs and benchmark follow-up that
  issue also scopes is not done here.
- `scripts/leakage_experiment.py` quantifies the library's central claim, which
  was previously architectural and untested. On a seeded donor-year panel across
  five seeds: walk-forward `FiscalYearGroupedSplitter` estimates the true future
  to within -0.014 ROC-AUC where a random `StratifiedKFold` is off by -0.030, so
  the splitter is worth roughly twice the accuracy in your estimate. Both CV runs
  exclude the year the target scores, because "train on everything earlier, score
  the final year" is what a walk-forward splitter's last fold does and leaving it
  in would hand walk-forward the win by construction. Computing the same aggregate
  features over the whole export instead of as of each panel year inflates the
  score by **+0.126 ROC-AUC**, under an identical model, splitter and label:
  roughly eight times what the splitter choice is worth. Correct feature timing is worth an order of magnitude more than a
  correct splitter, which is the case for freezing fit-time statistics and for the
  new `as_of` cutoff. Reported in `docs/explanation/benchmarks.md`, including the
  negative result: the common claim that a random split *inflates* a backtest did
  not reproduce here in three separate configurations. Closes #84.
- `.gitattributes` sets `CHANGELOG.md merge=union`. `AGENTS.md` requires every PR
  to add an entry under `## [Unreleased]`, so every concurrent PR conflicts with
  every other one, always in the same place and always additively. 10 of the last
  20 commits on `main` touch this file. `union` keeps both sides instead of
  stopping, which is the standard treatment for an append-only file. It is
  line-based rather than section-aware, and it never reports a conflict for this
  file at all: if two branches edit the same entry it keeps both, silently. So the
  `## [Unreleased]` block is worth a skim at release time, for a bullet under the
  wrong heading and for a duplicated one; `RELEASING.md` now says so. GitHub does not read
  `.gitattributes`, so its own merge behaviour is unchanged: the benefit is to the
  local `git merge origin/main` that currently absorbs the cost.
- `philanthropy.metrics.conformal_pvalue`: the non-smoothed split-conformal
- `philanthropy.metrics.conformal_pvalue`: the non-smoothed split-conformal
  p-value of a donor score against a held-out calibration set,
  `(1 + |{i : s_i >= s}|) / (n + 1)`. A calibrated probability threshold fixes no
  error rate; thresholding this p-value at `alpha` bounds the expected selection
  rate at `alpha` in finite samples with no distributional assumption. It is a
  selection-rate bound, not a false-positive rate: the latter reading needs a
  calibration set of nulls only, as in Bates et al. (2023). Both the `1 +`
  and the `+ 1` are load-bearing and tested: the result is never 0 and never
  above 1, and leave-one-out over exchangeable scores lands exactly on the
  uniform lattice.
- `as_of` on `EncounterTransformer` and `GratefulPatientFeaturizer`: an as-of
  cutoff that excludes encounters discharged after a given date from
  `encounter_summary_` at fit time. Without it there was no way to bound the
  encounter table to what was observable at the decision point, so a gift dated
  2020 was featurised from encounters recorded in 2024 and
  `days_since_last_discharge` was measured from the all-time max discharge. The
  failure was systematic rather than random: the more a donor engaged *after* the
  gift, the further the feature was pushed past the gift date and the more often
  it collapsed to `NaN`, destroying it for exactly the donors it should be
  strongest for. Defaults to `None`, which is the previous behaviour, so nothing
  changes until you opt in; set it to the last day of your training window for
  walk-forward evaluation.
- Test coverage for `constituent_events_to_features`: the all-unparseable-timestamps empty-frame path and the `distinct_source_systems` default-to-zero path when `sourceSystem` is absent from the input. (#51)
- `AGENTS.md`: every change, including maintainer- and agent-authored ones, must
  go on a branch and through a PR: no direct commits to `main`, no self-merges.
  go on a branch and through a PR, and never straight to `main`. (The blanket
  "no self-merges" this originally also promised is superseded below: with one
  account holding merge rights it could not hold.)
- `tests/test_no_network.py` enforces in CI what the docs now promise: the package
  makes **no network calls**. Every socket entry point is monkeypatched to raise,
  then a full train/score cycle, an imputation pass and a CiviCRM ingest all run.
  A telemetry hook, HTTP client or lazily downloaded asset added later fails this
  test instead of shipping.
- `docs/explanation/security_review_answers.md`: the ten questions an institutional
  security or privacy review actually asks, on one forwardable page (BAA status,
  dependency provenance, the pickle trust boundary, de-identification scope, bus
  factor, disclosure route).
- `make riskcov`: the risk-tier coverage floor as a single source of truth. `ci.yml`
  and `CONTRIBUTING.md` now both call it.
- `scripts/issue-drafts/_TEMPLATE.md` and `scripts/check_issue_lines.py`: the issue
  shape that converts, and a drift checker for the `path:line` references in issue
  bodies. Deliberately outside `.github/ISSUE_TEMPLATE/`, which is the public chooser.
- Two tests for guards that only fire at transform time and were previously
  uncovered: `MatchingGiftFeaturizer.transform` rejecting a non-DataFrame, and
  `ShareOfWalletScorer` enforcing the `capacity_col_idx` upper bound that `fit`
  deliberately does not check.

### Changed
- Logo: a new mark, an outlined heart crossed by a rising arrow, drawn as SVG so it
  stays crisp at favicon size and follows the colour scheme. `docs/assets/logo.svg`
  is the favicon, `overrides/.icons/philanthropy/heart-rise.svg` is inlined as the
  header logo, and `docs/assets/logo.png` is the regenerated wordmark lockup the
  README uses.
- Homepage figure: the affinity-score distribution is now a chart, not an ASCII dump
  of `describe()`, and it plots the held-out scores the quickstart reports after
  #89, not the in-sample ones. Interquartile bar, median notch, full min-to-max
  range, the overlapping tails left visible, and the 47-point separation between the
  two middle halves called out beside the held-out ROC-AUC of 0.932. The accent marks
  the group being ranked, muted ink the reference group; both fills clear 3:1 on
  their surface in each scheme. Hover gives the five-number summary and a
  collapsible table view carries every number, so nothing is gated behind the
  tooltip.
- Informational admonitions (note, info, tip, abstract, example, quote) now wear the
  palette instead of Material's blue; warning and danger keep their semantic colours.
- Documentation site: a new visual system (Fraunces display serif over Geist,
  a single amber accent, warm near-black canvas with a paper light mode),
  dark scheme first, and a homepage that shows the ten-line quickstart and its
  output above the fold. `mkdocs.yml` also gains section index pages, instant
  navigation, prev/next footer links, footer social links, and a correct
  `edit_uri` (the "edit this page" links previously pointed at a `master`
  branch that does not exist).
- Removed every em dash from the repository's prose, 442 of them across 91 files:
  `paper.md`, `README.md`, all documentation pages, docstrings, inline comments,
  `CHANGELOG.md`, the `Makefile`, `.flake8` and both CI workflows. Each site got
  the punctuation the sentence wanted, picked individually rather than by blanket
  substitution: a colon for a label or definition, commas for an appositive, a
  semicolon for two independent clauses, parentheses for a paired aside. The only
  em dash left is inside a nonprofit's name in
  `philanthropy/datasets/data/ciob_official_fundraising.csv`, which is source
  data rather than prose. No behaviour changes, though a handful of the edited
  sites are user-visible strings rather than prose: the pickle-trust warning in
  `philanthropy/cli.py` and several test comments.
- `AGENTS.md` said "never merge your own PR; open it and leave the merge to
  review" while `.github/CODEOWNERS` is `* @shivamlalakiya` and no second account
  holds merge rights. Taken literally the rule means nothing ever merges, and it
  was visibly not being followed. It now describes what is actually required: a
  PR for every change, green CI before merge, a second reviewer when one is
  available, the maintainer merging their own PR when one is not, and agents never
  merging at all. The section says explicitly that this is a description rather
  than an endorsement, and points at the real fix.
- Four docstrings described behaviour the code does not have, each now corrected
  against a test in `tests/test_documented_contracts.py`. `FiscalYearTransformer`
  said it *appends* `fiscal_year`/`fiscal_quarter`; `transform` in fact returns
  only those two columns and drops the input, which silently discarded a
  pipeline's features. `EncounterTransformer.fit` claimed it "prevents temporal
  data leakage"; it only guarantees that nothing from `X` enters the summary, and
  the summary itself has no as-of cutoff, so a 2020 gift is scored against 2024
  encounters. `WealthScreeningImputerKNN.group_col_idx` documented per-group
  stratified imputation "improving local accuracy"; it is stored and never read.
  The `GratefulPatientFeaturizer` service-line weights were attributed to
  "commonly-cited AMC development benchmarks"; they have no published source.
  No behaviour changed in this entry: the docs moved to meet the code.
- `FiscalYearGroupedSplitter` now documents the leakage it does **not** prevent:
  its grouping unit is the fiscal year, not the donor, so a donor with gifts in
  several fiscal years appears in both folds of a split. That is correct for a
  time-varying target and is leakage for a static per-donor label such as
  `is_major_donor`. The class docstring previously implied it prevented leakage
  generally.
- JOSS paper prep: restored the leakage (`kaufman2012leakage`, `kapoor2023leakage`)
  and grateful-patient-ethics (`collins2018grateful`) citations that the root
  `paper.md` had dropped, so the temporal-leakage claim in the Statement of need
  and the AMC domain claim both have sources again. Settled the author
  affiliation to "Independent Researcher" in `.zenodo.json`, which still said
  "Washington University in St. Louis" and so disagreed with `paper.md`; that
  value is minted into a permanent citable Zenodo record. Deleting the duplicate
  `paper/` draft and repointing `draft-pdf.yml` landed separately in #72.
- Added complete output-column documentation to all eleven preprocessing
  `get_feature_names_out` overrides that previously rendered blank in the API
  reference.
- Documented `fit`/`transform` on `CRMCleaner`, `FiscalYearTransformer` and
  `WealthPercentileTransformer`, including which attributes each `fit` freezes and
  that `WealthPercentileTransformer` ranks held-out rows against the frozen training
  distribution rather than the batch being transformed.
- Documented `fit`, `predict`, `predict_proba` and `predict_affinity_score` on
  `MovesManagementClassifier`, `PlannedGivingIntentScorer`, `MajorGiftClassifier` and
  `PropensityScorer`: the fitted attributes each sets, and that `PropensityScorer`'s
  default threshold returns `classes_[0]` for every row.
- `make_donor_dataset` now documents that it returns a **gift-level** frame, so
  `len(df) > n_donors` (each donor contributes 1–5 rows), and that
  `fiscal_year_start` and `lapse_rate` are accepted but currently unused.
- `CLAUDE.md` is now `AGENTS.md`, the tool-neutral convention, with `CLAUDE.md`
  reduced to an `@AGENTS.md` import. Agents other than Claude Code were reading no
  project instructions at all: not the leakage contract, not the dependency
  constraint, not `make ci`.
- `README.md` quickstart prints a result instead of ending in a bare `assert`, and
  documents the CLI path for readers who do not write Python.
- `SECURITY.md` supported-versions table named `0.5.x`, which has not been the
  installable release since `0.6.0`. Now `0.6.x`, and kept current by the
  `RELEASING.md` checklist. Adds GitHub private vulnerability reporting as the
  preferred disclosure channel.
- `AGENTS.md`'s merging section no longer bars agents from merging outright.
  An agent may now merge a PR under the same bar as the maintainer's own-PR
  merge (all required CI checks green, no second reviewer available), plus
  having actually read the diff and judged it good.
- `.gitignore` now excludes `.claude/CLAUDE.local.md`, for personal working
  notes that shouldn't end up in the repo.

### Fixed
- Version metadata now names the release that actually exists. `pyproject.toml`
  and `CITATION.cff` both declared `1.0.0`, which has no git tag, no PyPI
  artifact and no Zenodo deposit; PyPI's newest is `0.6.0` and so is the newest
  tag. `CITATION.cff` additionally dated that phantom 1.0.0 to `2026-08-01`,
  which is 0.6.0's release date, and its DOI comment cited the v0.6.0
  per-version DOI while the file claimed 1.0.0. Both now say `0.6.0`, and the
  comment states the rule: `version` tracks the newest **published** release,
  not `main`. `main` continues to carry unreleased `0.7.0` and `1.0.0` work, and
  those CHANGELOG headings stay `- TBD` until a release is cut. This also
  unblocks the JOSS archive step, which requires the submitted version to
  correspond to a real tagged, archived release. `RELEASING.md`'s "cutting a
  release `main` has already moved past" section assumed the tip carried the
  newest staged version and walked through tagging an older commit; with the
  tip back at the published version the normal branch-bump-date-tag path
  applies, so that section documents that instead and keeps the older-commit
  case as the caveat it is. Closes #88.
- **`generate_synthetic_donor_data` ran the domain's causal arrow backwards.**
  It drew `is_major_donor` from a logistic model of `years_active` and
  `event_attendance_count`, then drew `total_gift_amount` *conditional on that
  label*, so the strongest feature was generated from the answer. Measurably: a
  model given `total_gift_amount` scored ROC-AUC 0.935 against a causal Bayes
  accuracy ceiling of 0.768, beating the Bayes rate of the generator's own
  process by about 19 AUC points, which no model can legitimately do. Using
  cumulative lifetime giving to predict "is a major donor" is also the classic
  fundraising leakage this library exists to prevent, so the reference dataset
  was teaching the anti-pattern. `last_gift_date` was a second target-derived
  feature, drawn Beta for majors and uniform for everyone else.
  A latent giving capacity now drives everything: a confounder causing both the
  giving history and the label. `total_gift_amount` is a noisy realisation of
  capacity, `is_major_donor` a soft $25,000 threshold on it, and
  `last_gift_date` follows engagement. The model now sits **below** the ceiling
  (accuracy 0.759 against 0.806) rather than above it, which is the correct
  relationship. Held-out ROC-AUC moves from 0.935 to 0.814 and the base rate from
  0.687 to 0.378: worse numbers, trustworthy ones. Benchmark table, README
  quickstart and the benchmarks page are regenerated from the committed script.
  `generate_synthetic_donor_data` is Tier 1, so **this changes returned data for
  a documented-stable function**; release sequencing is the open version
  question. Closes #86.
- The README quickstart fitted and scored **the same rows**, then reported the
  resulting gap ("non-major donors top out at 39; no major donor scores below 65")
  as the headline result. That gap was a random forest reciting its training set:
  RF leaves go pure and `predict_affinity_score` is
  `predict_proba(X)[:, 1] * 100`. On held-out rows from the same 500-row sample
  the two groups overlap almost completely (non-major max 97.5, major min 6.5).
  The quickstart now splits before fitting and reports held-out ROC-AUC 0.932
  with overlapping score distributions, which is a weaker claim and a true one.
- `docs/explanation/benchmarks.md` distrusted its own numbers for the wrong
  reason. It said the synthetic data was "cleanly separable by construction"; the
  label is a Bernoulli draw with a real noise term and the irreducible error over
  the causal features is 23.2%. The actual problem is that the generator draws
  `total_gift_amount` **from** the label, so including that feature lets a model
  score ROC-AUC 0.935 against a causal Bayes ceiling of 0.768 accuracy: it beats
  the Bayes rate of its own data-generating process by about 19 AUC points, which
  is the signature of a target-derived feature. The page now measures and states
  this, and records that there is no validation on real donor data anywhere in the
  repository.
- Two `EncounterTransformer` output columns were documented with the wrong type
  and the wrong semantics. `days_since_last_discharge` was described as an
  "Integer number of days"; it is `float64`, and it has to be, because a donor
  absent from the encounter table gets `NaN` and an integer dtype cannot carry
  that. A caller who trusted the docstring and cast the column would silently
  destroy the missingness, which is signal in this library.
  `encounter_frequency_score` was described as a "Log-scaled count of distinct
  encounter records"; it is `log1p` of the **row** count, so a donor with three
  rows on two dates scores `log1p(3)`, not `log1p(2)`. Both are now stated
  correctly and locked by tests in `tests/test_documented_contracts.py`. Found
  during review of the em-dash branch; docstrings only, no behaviour change.
- `LapsePredictor` and `experimental.UpliftTLearner` validated input with
  `check_array`/`check_X_y` instead of `validate_data`, the convention every
  other estimator follows. Neither set `feature_names_in_`, so a DataFrame with
  reordered columns was silently scored instead of raising. These were the only
  two estimators in the package with that gap, and both are now closed with a
  regression test each.
- `CRMCleaner` NaN'd every value in a currency-formatted amount column, e.g.
  `"$1,000.00"` (the default export format for Raiser's Edge NXT and
  Salesforce NPSP), because `pd.to_numeric` treats the whole string as
  unparseable. It now strips currency symbols, thousands separators and
  parenthesised negatives before parsing, and raises rather than returning an
  all-NaN column when a column truly has nothing parseable in it. The
  string/numeric branch checks `pd.api.types.is_numeric_dtype` rather than
  `dtype == object`, so it also parses correctly under pandas 3.0's non-object
  default string dtype, not just the legacy `object` dtype.
- `MatchingGiftFeaturizer` ran zero `check_estimator` checks: `tags._skip_test =
  True` silently skipped the whole battery instead of excluding it from
  `_STANDARD_ESTIMATORS` with a documented reason, the way `RFMTransformer`
  already was. It has no such reason on its own (it genuinely cannot accept
  the generic numeric ndarrays the battery feeds), so this falsified the
  README/paper claim that every public estimator passes `check_estimator`.
  `FinancialForecastModel` had the same gap for no documented reason at all;
  it in fact passes the battery cleanly and is now in it. A new
  `test_every_public_estimator_is_covered_by_the_battery_or_documented` test
  cross-references `philanthropy.models.__all__` and
  `philanthropy.preprocessing.__all__` against `_STANDARD_ESTIMATORS` plus a
  reasoned exemption registry, so this can't recur silently. README, paper.md,
  and the design-principles/security-review docs now state the one real
  exception (`UpliftTLearner`) instead of claiming "every estimator" flatly.
- Two JOSS paper drafts were tracked at once: `paper.md`/`paper.bib` at the repo
  root (current, last touched 2026-08-11) and a stale copy in `paper/`
  (2026-08-01, different affiliation and bibliography style). `draft-pdf.yml`
  built only the stale one, so the current draft has never produced a PDF.
  Deleted `paper/`; the workflow now points at the root files.
- Saving a fitted `EncounterTransformer` or `GratefulPatientFeaturizer` wrote the
  **raw clinical encounter table into the model bundle**. Both take
  `encounter_df` as a constructor parameter, so `joblib.dump` / `save_model`
  persisted medical record numbers, attending physicians and service lines
  verbatim; a bundle attached to a ticket or handed to a vendor was a PHI
  disclosure. Both now drop the raw table on serialisation and keep only the
  per-donor `encounter_summary_` that `transform` actually reads, so a
  round-tripped transformer still scores identically. `clone` is unaffected
  (it goes through `get_params`, not pickle), and a refit now requires the table
  to be supplied again rather than reusing stale clinical rows. `SECURITY.md`
  previously treated pickles only as an inbound code-execution risk and never
  mentioned that a bundle you produce is itself donor data; it now does.
- `FiscalYearGroupedSplitter` never validated `n_splits`, despite documenting a
  `ValueError` for `n_splits < 1`. A non-positive value reached the
  `unique_fy[-(n_splits):]` slice, where it flips open-ended: `n_splits=0`
  yielded 3 folds on a 4-fiscal-year panel while `get_n_splits()` reported 0, and
  `n_splits=-1` yielded 3 while reporting -1. `cross_val_score` sizes its result
  array from `get_n_splits()`, so the two disagreeing is a real failure. Both
  entry points now validate through one helper, `gap_years < 0` and non-integer
  values are rejected, and a test asserts `get_n_splits() == len(list(split()))`
  across the parameter grid.
- **`donor_lifetime_value` overstated LTV whenever `retention_rate` was given.**
  It converted the retention rate to an expected lifespan, `L = 1 / (1 - r)`, and
  fed that mean into the concave annuity formula. By Jensen's inequality
  `NPV(E[L]) >= E[NPV(L)]`, so the result was biased high in one direction every
  time: +8.2% at `r = 0.8, d = 0.05` and +22.9% at `r = 0.9, d = 0.10`. A
  one-signed error does not average out across a portfolio, and this is a number
  that goes into board decks and acquisition-cost justifications. The retention
  branch now uses the correct closed form for a geometric lifetime,
  `E[NPV] = m / (1 + d - r)`, verified against a term-by-term expectation and a
  two-million-draw Monte Carlo. `retention_rate=1.0` with a positive discount
  rate now returns the perpetuity `m / d` rather than `inf`; it is still `inf`
  when `discount_rate` is 0. `retention_rate > 1` now raises instead of returning
  a negative number. The fixed-horizon path (`retention_rate=None`) is unchanged
  and was always correct, as is the `discount_rate=0` path in both modes, since an
  undiscounted sum is linear in the lifespan. **This changes returned values**:
  see `docs/explanation/fundraising_metrics.md` for both formulas and why they
  differ.
- `mkdocs.yml` had no `site_url`, so the generated `sitemap.xml` was empty and all
  38 documentation pages were uncrawlable, with no `rel=canonical` anywhere.
- `CONTRIBUTING.md` documented a risk-tier coverage command measuring `metrics/` and
  `model_selection/`, while CI measured `ingest/`, `cli.py` and `utils/_persistence.py`.
  A contributor could run the documented command, pass, and still fail CI on files it
  never looked at.

### Removed
- `FiscalYearGroupedSplitter._iter_test_indices` and `_iter_test_masks`. Both were
  unreachable (the class overrides `split`, so `cross_validate` never called
  either) and the comment claiming `BaseCrossValidator` requires them was false.

### Fixed
- `FiscalYearGroupedSplitter`'s module doctest asserted
  `... <= ... + 1 or True`, which passes for every possible input and so proved
  nothing about the split. It now asserts what the class actually promises,
  `fiscal_years[train_idx].max() < fiscal_years[test_idx].min()`, and a new
  `test_default_splitter_no_leakage_gap_years_zero` covers the default
  `gap_years=0` path, which had no leakage test at all. Thanks to
  [@fuleinist](https://github.com/fuleinist) (Chris Chen) for the first external
  contribution ([#30](https://github.com/PhilanthroPy-Project/PhilanthroPy/pull/30),
  closes [#26](https://github.com/PhilanthroPy-Project/PhilanthroPy/issues/26)).

### Added
- **CiviCRM contribution bridge**: `philanthropy.ingest.read_civicrm_contributions`
  and `civicrm_contributions_to_features` (Tier 2). Turns a CiviCRM contribution
  export, or an APIv4 `Contribution.get` result, into the one-row-per-donor
  feature table the estimators consume. Headers normalise to the APIv4 spelling,
  so the human export labels (`Contact ID`, `Total Amount`, `Contribution Date`)
  and the DB columns (`contact_id`, `total_amount`, `receive_date`) both work.

  It exists because two things a bare `pd.read_csv` gets wrong are expensive:
  CiviCRM writes payment-processor **test transactions** into the same table, and
  `contribution_status` separates `Completed` from `Pending`, `Failed`,
  `Refunded` and `Chargeback`. Test rows are always dropped; only `Completed` is
  counted unless `statuses` says otherwise, and asking for a status filter that
  cannot be applied warns instead of silently summing refunds.

  Recency is anchored to `reference_date` or the batch's latest gift, never a
  moving "now", the same leakage contract as the UniSchema bridge. See
  [docs/how-to/ingest_civicrm_contributions.md](docs/how-to/ingest_civicrm_contributions.md).

## [0.6.0] - 2026-08-01

### Breaking
- `pandas>=2.0` is now the declared floor (was `>=1.5`). The ingest bridge pins
  `format="ISO8601"`, which is pandas 2.0+; on a conforming 1.5.x install
  `errors="coerce"` silently produced an all-NaT, zero-row feature frame. The
  build backend floor moves to `setuptools>=77` for the PEP 639 license fields.
- `DischargeToSolicitationWindowTransformer.transform` now raises `ValueError`
  when given a DataFrame without `days_since_discharge_col`, instead of reading
  `X.iloc[:, 0]`. That fallback made a serial `Pipeline` behind
  `FiscalYearTransformer` score every donor `0.0` and exit cleanly. Route the
  transformer with a `ColumnTransformer`; see
  `docs/tutorials/building_your_first_model.md` for the migration.
- `philanthropy.experimental.LapsePredictor` is **removed**. It collided by name
  with `philanthropy.models.LapsePredictor` and took different positional
  arguments. Tier 3, so no deprecation runway. Use `models.LapsePredictor`.
- `PropensityScorer.predict` now uses a strict threshold comparison
  (`proba > threshold`), flipping its default-threshold prediction from class 1
  to class 0. scikit-learn requires `argmax(predict_proba) == predict`, and
  `argmax` of a tied `[0.5, 0.5]` row is index 0. Arbitrary either way for a
  constant scorer; only its ROC-AUC of 0.500 ever carried information.
- `PropensityScorer.fit` now raises `ValueError` on a multiclass `y`.

### Added
- `philanthropy.model_selection`, `.experimental` and `.visualisation` are now
  importable from `import philanthropy` and listed in `__all__`; they raised
  `AttributeError` before while the docs rendered reference pages for them.
- `philanthropy/py.typed`: the package now ships its type information.
- `joblib>=1.2` is declared; it was a direct import carried transitively.
- `tests/test_public_api_contract.py`: an executable spec for the public API:
  subpackage `__all__` completeness, reference-page coverage, the
  `predict_<thing>_(score|forecast)` naming and shape contract, and
  `get_feature_names_out` width. Two named exemptions, each with a reason.
- `tests/test_metrics_oracles.py`: closed-form oracles for the money metrics:
  the textbook Gini definition, a term-by-term discounted annuity, and the
  EEOC four-fifths worked example.
- Seven how-to guides: use the CLI, ingest UniSchema events, recommend ask
  amounts, score matching-gift eligibility, measure campaign efficiency, audit
  score fairness, estimate appeal uplift. Reference pages for `experimental`,
  `utils` and `cli`. A stability-tier and score-scale table in
  `docs/reference/index.md`.
- `.zenodo.json` and a concept-DOI placeholder in `CITATION.cff`.

### Deprecated
All of the following still work and emit `DeprecationWarning`. **Removed in
0.7.0.**

| Deprecated | Use instead |
|---|---|
| `AskAmountRecommender.predict_ask_array` | `ask_ladder` |
| `ShareOfWalletRegressor.predict_capacity_ratio` | `capacity_ratio` |
| `MovesManagementClassifier.predict_action_priority` | `action_priority` |
| `PlannedGivingIntentScorer.predict_bequest_intent_score` | `predict_intent_score` |

The `predict_` prefix is now reserved for methods that take X alone and return
one value per row. The first three returned a `(n, 3)` dollar matrix, required a
second argument, and returned a dict respectively.

Three constructor parameters have no effect and warn when set to a non-default
value: `LapsePredictor(lapse_window_years=...)`,
`PropensityScorer(estimator=...)`,
`FiscalYearGroupedSplitter(fiscal_year_start=...)`. All removed in 0.7.0.

### Fixed
- `philanthropy.__version__` is read from installed metadata. It reported
  `0.4.0` against a `0.5.0` package, and every bundle written by `save_model`
  carried the wrong stamp.
- `MajorGiftClassifier.n_iter_` reports the real mean boosting iterations across
  the calibration folds instead of a hardcoded `1`.
- `GratefulPatientFeaturizer.transform` emits a `UserWarning` before each
  all-zero fallback instead of silently returning `zeros((n, 4))`.
- `philanthropy train --features " , "` now exits with an error instead of
  fitting on a zero-column matrix.
- `read_constituent_events` raises `FileNotFoundError` for a missing path
  instead of surfacing an opaque OSError from the single-file branch.
- `WealthScreeningImputer` no longer emits a `Mean of empty slice`
  `RuntimeWarning` on an all-NaN column.
- `CRMCleaner.fiscal_year_start` is documented as validated-but-unused.
- Doc corrections: `ShareOfWalletScorer` capacity-tier thresholds, PHI-dropping
  attributed to `EncounterTransformer` rather than `CRMCleaner`, and the
  `GratefulPatientFeaturizer` output columns.

### Changed
- The `check_estimator` battery is consolidated into one list in
  `tests/test_sklearn_compliance.py`. `MajorGiftClassifier` runs at
  `max_iter=10`, cutting suite runtime by roughly two thirds;
  `PropensityScorer`, `WealthPercentileTransformer`,
  `WealthScreeningImputerKNN`, `ShareOfWalletScorer`, `CRMCleaner`,
  `EncounterRecencyTransformer` and bare-default variants were added.
  `RFMTransformer` moved to an explicit contract class: its `_skip_test=True`
  tag was running 1 check instead of 46.
- Branch coverage is enabled and gated; the risk-tier subtree has its own floor.
- `docs/explanation/benchmarks.md` reports mean and min–max across five seeds
  instead of three decimals from one split.
- CI: the duplicate full-suite run is gone, lint runs once instead of five
  times, and there are new `floors` (lowest-direct dependency resolution),
  `package`, and `minimal` (no-matplotlib import) jobs plus a macOS leg.
- `publish.yml` gates on tag ↔ `pyproject.toml` ↔ `CHANGELOG.md` agreement, and
  both third-party actions are SHA-pinned.

## [0.5.0] - 2026-07-24
### Added
- Donor-base concentration metrics: `gift_concentration_gini` and
  `top_donor_share` (`philanthropy.metrics`).
- Campaign-efficiency metrics: `cost_per_dollar_raised` and `fundraising_roi`.
- `philanthropy.utils.save_model` / `load_model`: self-describing model
  bundles that warn on a scikit-learn / PhilanthroPy version mismatch at load
  time; the CLI now persists and loads through them.
- `AskAmountRecommender`: capacity regressor exposing a discrete gift-array
  ask ladder (`predict_ask_array`).
- `MatchingGiftFeaturizer`: corporate matching-gift features (`has_employer`,
  `match_ratio`, `potential_matched_amount`), leakage-safe.
- `philanthropy.experimental.UpliftTLearner`: two-model uplift (treatment-
  effect) scorer for appeals (`predict_uplift_score`).
- `MovesManagementClassifier` is now covered by the `check_estimator` battery.
- Dependabot (`github-actions` + `pip`), a CodeQL scanning workflow, and a
  `.pre-commit-config.yaml` running the flake8 / mypy gates locally.
- `philanthropy.ingest` and `philanthropy.visualisation` API reference pages.
- `constituent_events_to_features` carries `first_name` / `last_name` through to
  the donor feature table when the UniSchema feed supplies them (guarded; null
  when absent).
- Community health files: `.github` issue/PR templates,
  `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1), and `SECURITY.md`.
- flake8 lint gate: `.flake8` (enforces pyflakes `F` + syntax `E9` defects),
  a `make lint` target folded into `make ci`, and a CI step.
- `philanthropy.inspection.donor_feature_importance`: model-agnostic permutation
  feature importance (dependency-free interpretability; works on calibrated models
  that lack `feature_importances_`).
- `philanthropy.metrics.disparate_impact_ratio` and `selection_rate_by_group`:
  four-fifths-rule fairness diagnostics for scored cohorts.
- `philanthropy` command-line interface (`train` / `score` / `validate`) over CSV.
- `EncounterTransformer(pii_patterns=...)` to override the PII column heuristic
  (defaults broadened); `allow_negative_days=True` now emits a compliance `UserWarning`.
- `GratefulPatientFeaturizer(capacity_weights=...)` to override service-line weights.
- mypy type-check gate wired into `make ci` and CI.
- Docs: Responsible Use & Compliance, Model Validation & Benchmarks, vendor
  comparison, and model-persistence guides; `.github/CODEOWNERS`; a JOSS `paper/`.

### Security
- CLI `score` neutralizes spreadsheet formula-injection (CWE-1236) in
  donor-controlled string cells before writing the output CSV.
- Documented the model-bundle pickle trust boundary in `SECURITY.md` and the
  `score` / `validate` `--help` text.
- `read_constituent_events` skips symlinked files (path-traversal hardening)
  and reports malformed JSON with the offending file and line number.
- Least-privilege `permissions: contents: read` on all GitHub Actions
  workflows.

### Fixed
- `RFMTransformer` freezes the recency reference date in `fit`
  (`reference_date_`) instead of recomputing it from the transform batch, a
  leakage-contract violation that made a donor's recency depend on batchmates.
- `LapsePredictor.predict_lapse_score` no longer raises `IndexError` when fit
  on a single-class training fold.
- `MovesManagementClassifier` rejects continuous targets and exposes `n_iter_`.
- `disparate_impact_ratio` / `selection_rate_by_group` raise on missing
  (`NaN`/`None`) group labels instead of silently returning `NaN`.
- Removed dead code (`_assign_tier` / `_TIER_THRESHOLDS`, `_resolve_cols`,
  redundant `_more_tags`) and cleared 4 pre-existing type-check errors.
- Cleared 31 real-defect lint violations (unused imports/variables) across the
  package and tests, including two dead code blocks.
- Corrected `EncounterTransformer` API drift in the grateful-patient tutorial and
  the README example (invalid `encounter_date_col` / `donor_id_col` kwargs →
  `discharge_col` / `merge_key`; pipeline scored via `predict_proba`).
- README metrics table listed `retention_rate`; the real export is
  `donor_retention_rate`.
- Removed a documented-but-nonexistent `fiscal_year_start` parameter from the
  `EncounterTransformer` and `WealthScreeningImputer` docstrings.

## [0.4.0] - 2026-07-18
### Added
- `philanthropy.ingest`: the UniSchema on-ramp. `constituent_events_to_features()`
  aggregates a UniSchema `ConstituentEvent` stream into a one-row-per-donor
  feature table whose columns (`total_gift_amount`, `years_active`,
  `event_attendance_count`, `last_gift_date`, ...) feed the estimators directly;
  `read_constituent_events()` loads UniSchema's JSON / NDJSON egress files.
  Leakage-safe (recency anchored to an explicit `reference_date` or the batch's
  latest event), at-least-once-safe (deduplicates by `eventId`).
- `constituent_events_to_features` and `read_constituent_events` re-exported at
  the top level (`from philanthropy import constituent_events_to_features`).
- `examples/quickstart.py` and `examples/unischema_to_scores.py`: runnable,
  end-to-end scripts (train + score; UniSchema `ConstituentEvent` stream →
  features → score). Smoke-tested in `tests/test_examples.py`.
- tests/test_ingest.py (aggregation, identity resolution, dedup, file/dir
  readers, mixed-currency warning, estimator integration)

### Fixed
- Pinned `scikit-learn>=1.6`; the code relies on `validate_data` and
  `__sklearn_tags__`, both 1.6+ APIs, so an unpinned install on 1.3–1.5
  imported broken.
- `MovesManagementClassifier` now imports on Python 3.9 (added
  `from __future__ import annotations`; its `str | dict | None` annotation
  was evaluated eagerly and crashed the advertised 3.9).
- Removed the nonexistent `philanthropy==0.2.0` pin from `environment.yml`
  that made `conda env create` fail.
- `constituent_events_to_features` warns on a mixed-currency batch instead of
  silently summing unlike amounts into `total_gift_amount`.
- `EncounterRecencyTransformer` no longer raises `OverflowError` when two
  encounter dates span more than ~292 years (a `datetime64[ns]` timedelta
  overflows int64); it falls back to day-resolution differencing.

### Changed
- README leads installation with `pip install philanthropy`; fixed the Tests
  badge and the UniSchema scoring snippet.
- Sharpened the PyPI `description`, added `machine-learning` /
  `predictive-analytics` / `data-science` / `python` keywords, and added the
  UniSchema project URL (pyproject + CITATION.cff).
- README roadmap corrected (docs site, PyPI, and retention-waterfall plot moved
  to Completed); dropped the stale per-file test table; ingest docs/example now
  point at UniSchema's real `data/egress/` path.
- `PropensityScorer` documented as a constant P=0.5 baseline (points to
  `DonorPropensityModel`); added docstrings for the metrics helpers and
  `predict_action_priority`; `CONTRIBUTING.md` gained a Setup section.

## [0.3.0] - 2026-07-17
### Added
- FinancialForecastModel: hybrid LSTM-ARIMA revenue/giving forecaster
  (linear ARIMA-surrogate + neural residual component) with
  `predict_revenue_forecast(X, horizon)`; leakage-safe: fill values and
  autoregressive coefficients frozen at `fit()`; passes sklearn
  `check_estimator`
- tests/test_forecast_model.py (fit/predict, forecast horizon, leakage,
  NaN handling, check_estimator compliance)
- PyPI packaging: complete project metadata, classifiers, keywords, and
  project URLs (docs / repo / changelog / issues); version bumped to 0.3.0
- MANIFEST.in so the sdist ships source only (no tests/dev artifacts)
- PyPI Trusted Publishing workflow (.github/workflows/publish.yml): OIDC,
  no stored token, fires on published GitHub Releases (v*.*.*)
- CONTRIBUTING.md split out of the README
- CITATION.cff for Zenodo/DOI archival
- README "Research" section mapping the literature to concrete estimators,
  and an affinity-distribution visual

## [0.2.0] - 2026-03-14
### Added
- GitHub Actions CI workflow (Python 3.10 + 3.11 matrix)
- Coverage gate: pytest --cov-fail-under=85
- Makefile with check / test / coverage / ci targets
- Branch protection + PR-based merge workflow
- DischargeToSolicitationWindowTransformer (2-column output: in_window, window_position_score)
- PlannedGivingIntentScorer with predict_intent_score()
- LapsePredictor: production RF, predict_lapse_score(), full param set
- 1052 tests across 23 test files (up from 161 across 7)
- Coverage: 88.29%

### Fixed
- SolicitationWindowTransformer.transform() now returns (n, 2) not (n, 3)
- Removed contradictory test_output_shape_is_n_by_3
- InvalidParameterError accepted alongside ValueError (sklearn 1.6+ compat)
- check_do_not_raise_errors_in_init_or_set_params: validation moved to fit()
- Hypothesis tests stabilised with @settings(suppress_health_check=...)

## [0.1.0] - 2026-01-01
### Added
- Initial release: DonorPropensityModel, ShareOfWalletRegressor,
  MajorGiftClassifier, CRMCleaner, WealthScreeningImputer,
  FiscalYearTransformer, EncounterTransformer, RFMTransformer
- philanthropy.metrics: donor_retention_rate, donor_acquisition_cost,
  donor_lifetime_value
- philanthropy.visualisation: plot_affinity_distribution
- philanthropy.utils: make_donor_dataset
- 161 tests across 7 test files
