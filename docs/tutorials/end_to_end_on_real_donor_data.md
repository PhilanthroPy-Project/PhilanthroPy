---
description: "The whole PhilanthroPy path on 95,412 real donors from KDD Cup 1998: ingest, as-of features, leakage, response and gift-size models, the mailing decision in dollars, and a disparity check."
---

# End to End on Real Donor Data

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/notebooks/04_kdd98_end_to_end.ipynb)

The other tutorials use synthetic data. This one runs on a real donor file:
**KDD Cup 1998**, 95,412 donors with the gift history of 22 earlier mailings and
the outcome of one more. The notebook is
[`examples/notebooks/04_kdd98_end_to_end.ipynb`](https://github.com/PhilanthroPy-Project/PhilanthroPy/blob/main/examples/notebooks/04_kdd98_end_to_end.ipynb);
open it in Colab with the badge above, or run it locally in about a minute.
The first run downloads the dataset (about 35 MB) once to `~/philanthropy_data`.

## What it covers

| Step | What you do | PhilanthroPy pieces |
|---|---|---|
| 1. Size up the file | Concentration, response rate, average gift | `gift_concentration_gini`, `top_donor_share` |
| 2. Clean gift log | Reshape a wide export into one row per gift, clean it, add fiscal years; retention and lifetime value | `CRMCleaner`, `FiscalYearTransformer`, `donor_retention_rate`, `donor_lifetime_value`, `plot_retention_waterfall` |
| 3. CRM export | Read the gifts back as a Raiser's Edge export, with a pledge row that a naive sum would double-count | `read_raisers_edge_gifts`, `raisers_edge_gifts_to_features` |
| 4. As-of features | Recency, frequency, monetary value and tenure as of the mailing date | `RFMTransformer(as_of=...)` |
| 5. Leakage | Backtest a lapse model with features built as of each period against the same features built over the whole file | `LapsePredictor`, `FiscalYearGroupedSplitter` |
| 6. Response model | Who will respond, with a leakage-safe wealth-screen imputer, and why they score high | `WealthScreeningImputer`, `MajorGiftClassifier`, `plot_affinity_distribution`, `donor_feature_importance` |
| 7. Gift size | Predicted gift with a calibrated 90% interval, and an ask ladder | `AskAmountRecommender`, `GiftIntervalCalibrator`, `interval_report` |
| 8. Mailing decision | Mail when expected gift beats the $0.68 cost, compared against mailing everyone | `fundraising_roi`, `cost_per_dollar_raised` |
| 9. Disparity check | Selection rate by recorded gender, four-fifths rule | `selection_rate_by_group`, `disparate_impact_ratio` |
| 10. Save | Store the fitted pipeline with its features and versions | `save_model`, `load_model` |

## What the notebook finds

From one run (`random_state=0`, 30% of donors held out):

- **Leakage inflates the backtest.** On a 20,000-donor sample, a walk-forward
  backtest of the lapse model reads ROC-AUC 0.717 with as-of features and 0.801
  when the same totals are built over the whole file. On the held-out final
  mailing the whole-history backtest overpromises by 0.218, against 0.170 for
  the as-of one. The full experiment on all 95,412 donors is in
  [Real-data replication](../explanation/real_data_replication.md).
- **Response model:** ROC-AUC 0.613 on held-out donors. The strongest signals
  are last gift, largest gift and number of gifts.
- **Gift-size interval:** certified 90.1% coverage; 88.7% observed on the test
  responders.
- **Mailing decision:** model-targeted mailing sends 18,109 pieces instead of
  28,624 and nets $3,877 instead of $3,035 on the held-out donors.
- **Disparity:** selection rates of 0.62 (F) and 0.66 (M), a ratio of 0.94,
  above the usual 0.8 flag.

The Raiser's Edge reader in step 3 is newer than the 0.7.1 release; until the
next release, install from GitHub (the first cell shows how).

!!! note "Dataset terms"
    Under the KDD Cup 1998 terms, teaching material must not name the
    organisation that supplied the data. Cite it only as "KDD Cup 1998".
