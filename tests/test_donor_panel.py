"""
tests/test_donor_panel.py
=========================
Unit tests for philanthropy.datasets.make_donor_panel.

The generator exists so that the transformers a cross-sectional frame cannot
reach (RFM, walk-forward splitting, ``as_of`` cutoffs, grateful-patient
featurization) have something to run on. So the tests here check both halves:
the invariants of the panel itself, and that those transformers actually
consume it.
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.datasets import make_donor_panel
from philanthropy.model_selection import FiscalYearGroupedSplitter
from philanthropy.preprocessing import EncounterTransformer, RFMTransformer

GIFT_COLUMNS = ["donor_id", "gift_date", "gift_amount", "fiscal_year", "appeal"]
DONOR_COLUMNS = ["donor_id", "first_gift_fy", "wealth_estimate", "employer"]
ENCOUNTER_COLUMNS = ["donor_id", "admit_date", "discharge_date", "service_line"]


# ---------------------------------------------------------------------------
# Schema and shape
# ---------------------------------------------------------------------------


def test_default_keys_are_gifts_and_donors():
    panel = make_donor_panel(n_donors=100, random_state=0)
    assert sorted(panel) == ["donors", "gifts"]


def test_column_names_and_dtypes():
    panel = make_donor_panel(n_donors=100, random_state=0)
    gifts, donors = panel["gifts"], panel["donors"]

    assert list(gifts.columns) == GIFT_COLUMNS
    assert list(donors.columns) == DONOR_COLUMNS
    assert pd.api.types.is_datetime64_any_dtype(gifts["gift_date"])
    assert pd.api.types.is_float_dtype(gifts["gift_amount"])
    assert pd.api.types.is_integer_dtype(gifts["fiscal_year"])


def test_one_row_per_donor_in_the_donor_table():
    panel = make_donor_panel(n_donors=250, random_state=1)
    donors = panel["donors"]
    assert len(donors) == 250
    assert donors["donor_id"].is_unique
    assert list(donors["donor_id"]) == list(range(250))


def test_fiscal_years_span_the_requested_range():
    panel = make_donor_panel(
        n_donors=400, n_years=4, start_fiscal_year=2030, random_state=2
    )
    assert set(panel["gifts"]["fiscal_year"]) == {2030, 2031, 2032, 2033}


# ---------------------------------------------------------------------------
# Panel invariants
# ---------------------------------------------------------------------------


def test_at_most_one_gift_per_donor_year():
    """The documented contract, and what makes 'recent' well defined."""
    gifts = make_donor_panel(n_donors=500, random_state=3)["gifts"]
    assert not gifts.duplicated(["donor_id", "fiscal_year"]).any()


def test_gift_dates_fall_inside_their_own_fiscal_year():
    """Fiscal year N runs 1 July N-1 to 30 June N."""
    gifts = make_donor_panel(n_donors=500, random_state=4)["gifts"]
    opens = pd.to_datetime(
        [f"{fy - 1}-07-01" for fy in gifts["fiscal_year"]]
    )
    closes = pd.to_datetime([f"{fy}-06-30" for fy in gifts["fiscal_year"]])
    assert (gifts["gift_date"] >= opens).all()
    assert (gifts["gift_date"] <= closes).all()


def test_gifts_are_sorted_by_date():
    gifts = make_donor_panel(n_donors=300, random_state=5)["gifts"]
    assert gifts["gift_date"].is_monotonic_increasing


def test_gift_amounts_are_positive():
    """A row exists only where a gift happened, so zero is a bug, not a lapse."""
    gifts = make_donor_panel(n_donors=500, random_state=6)["gifts"]
    assert (gifts["gift_amount"] > 0).all()


def test_first_gift_fy_agrees_with_the_gift_table():
    panel = make_donor_panel(n_donors=400, random_state=7)
    observed = panel["gifts"].groupby("donor_id")["fiscal_year"].min()
    stated = panel["donors"].set_index("donor_id")["first_gift_fy"]

    assert (stated.loc[observed.index] == observed).all()
    never_gave = stated.index.difference(observed.index)
    assert stated.loc[never_gave].isna().all()


def test_wealth_estimate_is_partly_missing():
    """WealthScreeningImputer has nothing to do on a fully-populated column."""
    donors = make_donor_panel(n_donors=1000, random_state=8)["donors"]
    missing = donors["wealth_estimate"].isna().mean()
    assert 0.2 < missing < 0.4


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_same_seed_reproduces_every_frame():
    a = make_donor_panel(n_donors=200, include_encounters=True, random_state=9)
    b = make_donor_panel(n_donors=200, include_encounters=True, random_state=9)
    assert sorted(a) == sorted(b)
    for key in a:
        pd.testing.assert_frame_equal(a[key], b[key])


def test_different_seeds_give_different_panels():
    a = make_donor_panel(n_donors=200, random_state=10)["gifts"]
    b = make_donor_panel(n_donors=200, random_state=11)["gifts"]
    assert len(a) != len(b) or not a.equals(b)


def test_no_seed_still_produces_a_valid_panel():
    panel = make_donor_panel(n_donors=50)
    assert list(panel["gifts"].columns) == GIFT_COLUMNS
    assert len(panel["donors"]) == 50


def test_asking_for_encounters_does_not_change_the_giving_process():
    """Encounter draws happen after the giving draws, and must stay there."""
    without = make_donor_panel(n_donors=300, random_state=12)
    with_enc = make_donor_panel(
        n_donors=300, include_encounters=True, random_state=12
    )
    pd.testing.assert_frame_equal(without["gifts"], with_enc["gifts"])
    pd.testing.assert_frame_equal(without["donors"], with_enc["donors"])


# ---------------------------------------------------------------------------
# Encounters
# ---------------------------------------------------------------------------


def test_encounters_absent_unless_requested():
    assert "encounters" not in make_donor_panel(n_donors=100, random_state=13)


def test_encounter_schema_and_ordering():
    enc = make_donor_panel(
        n_donors=400, include_encounters=True, random_state=14
    )["encounters"]
    assert list(enc.columns) == ENCOUNTER_COLUMNS
    assert (enc["discharge_date"] > enc["admit_date"]).all()
    assert enc["donor_id"].is_monotonic_increasing


def test_only_some_donors_have_encounters():
    panel = make_donor_panel(
        n_donors=1000, include_encounters=True, random_state=15
    )
    patients = panel["encounters"]["donor_id"].nunique()
    assert 0 < patients < 1000
    assert set(panel["encounters"]["donor_id"]).issubset(
        set(panel["donors"]["donor_id"])
    )


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_donors", [0, -1])
def test_rejects_empty_donor_pool(n_donors):
    with pytest.raises(ValueError, match="n_donors must be at least 1"):
        make_donor_panel(n_donors=n_donors)


@pytest.mark.parametrize("n_years", [1, 0, -3])
def test_rejects_a_panel_too_short_to_label(n_years):
    with pytest.raises(ValueError, match="n_years must be at least 2"):
        make_donor_panel(n_donors=10, n_years=n_years)


# ---------------------------------------------------------------------------
# The reason this generator exists: the transformers a flat frame cannot reach
# ---------------------------------------------------------------------------


def test_gifts_feed_rfm_transformer_without_renaming():
    panel = make_donor_panel(n_donors=300, random_state=16)
    rfm = RFMTransformer().fit_transform(panel["gifts"])
    assert list(rfm.columns) == ["donor_id", "recency", "frequency", "monetary"]
    assert len(rfm) == panel["gifts"]["donor_id"].nunique()


def test_fiscal_years_feed_the_walk_forward_splitter():
    gifts = make_donor_panel(n_donors=300, n_years=5, random_state=17)["gifts"]
    groups = gifts["fiscal_year"].to_numpy()
    X = gifts[["gift_amount"]].to_numpy()

    splitter = FiscalYearGroupedSplitter(n_splits=3, drop_repeat_donors=False)
    folds = list(splitter.split(X, groups=groups))
    assert len(folds) == 3
    for train_idx, test_idx in folds:
        assert groups[train_idx].max() < groups[test_idx].min()


def test_encounters_feed_the_encounter_transformer_with_an_as_of_cutoff():
    panel = make_donor_panel(
        n_donors=300, include_encounters=True, random_state=18
    )
    gifts = panel["gifts"][["donor_id", "gift_date", "gift_amount"]].copy()
    cutoff = gifts["gift_date"].max()
    # Dates as strings, exactly as EncounterTransformer's own documented
    # example passes them. A real datetime64 column raises DTypePromotionError
    # inside sklearn's validation; that is a wart in the transformer, not in
    # this generator, and it is tracked separately.
    gifts["gift_date"] = gifts["gift_date"].dt.strftime("%Y-%m-%d")

    transformer = EncounterTransformer(
        encounter_df=panel["encounters"],
        discharge_col="discharge_date",
        gift_date_col="gift_date",
        merge_key="donor_id",
        as_of=cutoff,
    )
    transformer.set_output(transform="pandas")
    out = transformer.fit_transform(gifts)

    assert "days_since_last_discharge" in out.columns
    assert len(out) == len(gifts)


def test_a_derived_label_is_a_plain_set_membership_check():
    """The docstring tells users to derive labels this way; keep it true."""
    gifts = make_donor_panel(n_donors=200, n_years=3, random_state=19)["gifts"]
    gave = set(zip(gifts["donor_id"], gifts["fiscal_year"]))

    donor, fy = gifts.iloc[0][["donor_id", "fiscal_year"]]
    assert (donor, fy) in gave
    assert np.isscalar(int((donor, fy) in gave))
