"""Every lapse benchmark feeds LapsePredictor the same gift columns (E.15
3.6): the model's verdict should be about the model, not about which bench
wrote its features. Each file may add columns it alone has (PSID household
income, wealth, volunteering), but the shared builder's core columns must be
in every lapse bench's feature set.

Runs on small stand-in tables shaped like each file, so it needs neither
the DonorsChoose nor the PSID download.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from philanthropy.datasets import make_donor_panel
from philanthropy.ingest._snapshots import CORE_COLUMNS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import benchmark_models_vs_baselines as bm  # noqa: E402


def _donorschoose_like():
    gifts = make_donor_panel(n_donors=200, n_years=6, random_state=0)["gifts"]
    return gifts[["donor_id", "gift_date", "gift_amount"]]


def _psid_like_long_table():
    rng = np.random.default_rng(0)
    rows = []
    for hh in range(150):
        for year in (2011, 2013, 2015, 2017):
            if rng.random() < 0.1:
                continue
            cats = {c: float(rng.choice([0, 0, 50, 200])) for c in ("giving_religious", "giving_needy", "giving_other")}
            rows.append({
                "household_key": f"h{hh}", "year": year, "total_giving": sum(cats.values()), **cats,
                "itemized_charitable_contrib_amount": 0.0, "family_income": float(rng.integers(20, 200)) * 1000,
                "wealth1": 1e4, "wealth2": 2e4, "head_volunteer_hours_annual": 0.0,
                "spouse_volunteer_hours_annual": 0.0, "household_volunteer_hours_regular": 0.0,
                "head_volunteer_hours_typical_week": 0.0, "spouse_volunteer_hours_typical_week": 0.0,
            })
    return pd.DataFrame(rows)


def _assert_core(snap, id_col):
    assert not snap.empty
    missing = set(CORE_COLUMNS) - set(bm._snapshot_feature_cols(snap, id_col))
    assert not missing, f"lapse bench is missing shared columns {sorted(missing)}"
    # Multi-year donors only: everyone gave in T and in some earlier period.
    assert (snap["periods_since_first_gift"] >= bm.LAPSE_MIN_YEARS_GIVEN - 1).all()


@pytest.mark.parametrize("momentum", [False, True])
def test_donorschoose_lapse_uses_shared_columns(momentum):
    _assert_core(bm._donorschoose_lapse_snapshots(_donorschoose_like(), momentum), "donor_id")


def test_psid_lapse_uses_shared_columns(monkeypatch):
    long_df = _psid_like_long_table()
    monkeypatch.setattr(bm, "_psid_long_table", lambda *_: long_df)
    snap = bm._psid_lapse_snapshots("unused", "unused", False)
    _assert_core(snap, "household_key")
    assert "family_income" in snap.columns, "PSID keeps its own household columns next to the shared ones"


@pytest.mark.skip(reason=(
    "KDD Cup 1998 lapse stays on its promotion panel: the gift history is effectively unrecorded for the "
    "last four promotions (242 donors gave in RAMNT_3, against 4,843 to the 97NK mailing and 8,000 to 27,000 "
    "in earlier promotions), so 'gave in T' at the published test period is not a real population"
))
def test_kdd98_lapse_uses_shared_columns():
    pass


@pytest.mark.skip(reason="synthetic lapse moves onto the shared builder with the synthetic panel")
def test_synthetic_lapse_uses_shared_columns():
    pass


def test_upgrade_transfer_scores_psid_on_columns_it_never_fit(monkeypatch):
    gifts = make_donor_panel(n_donors=400, n_years=6, random_state=0)["gifts"][["donor_id", "gift_date", "gift_amount"]]
    monkeypatch.setattr(bm, "_donorschoose_gift_log", lambda *_: gifts)
    long_df = _psid_like_long_table()
    long_df["total_giving"] *= 5  # some households reach the $100-999 band and cross $1,000
    monkeypatch.setattr(bm, "_psid_long_table", lambda *_: long_df)
    out = bm.bench_upgrade_transfer("unused", "unused", "unused", n_test_folds=2)
    assert set(out) == {"psid"} and out["psid"]
    assert all(r.dataset == "psid" and "transfer" in r.model for r in out["psid"])
