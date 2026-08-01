"""Regression guards for preprocessing transformer leakage / frozen state.

The transformers under test are currently CORRECT; these tests are regression
insurance that locks in their leakage-safety contract (fitted statistics are
computed from TRAINING data in ``fit``, FROZEN, and never recomputed per
transform batch; ``transform`` is idempotent). Each test verifies a property
the design guarantees today so a future refactor that reintroduces batch
dependence or mutation leakage fails loudly.
"""

import numpy as np
import pandas as pd

from philanthropy.preprocessing._encounter_recency import (
    EncounterRecencyTransformer,
)
from philanthropy.preprocessing._matching_gift import (
    MatchingGiftFeaturizer,
)
from philanthropy.preprocessing._grateful_patient import (
    GratefulPatientFeaturizer,
)
from philanthropy.preprocessing._share_of_wallet import (
    ShareOfWalletScorer,
    WealthScreeningImputerKNN,
)
from philanthropy.preprocessing._wealth_percentile import (
    WealthPercentileTransformer,
)


# ---------------------------------------------------------------------------
# 1. GratefulPatientFeaturizer
# ---------------------------------------------------------------------------

_ENC_DF = pd.DataFrame({
    "donor_id": [1, 2, 3],
    "discharge_date": ["2022-01-01", "2022-06-15", "2022-11-30"],
    "service_line": ["cardiac", "oncology", "general"],
    "attending_physician_id": ["P1", "P2", "P3"],
})
_DONOR_X = pd.DataFrame({"donor_id": [1, 2, 3]})


def test_grateful_patient_summary_frozen_against_source_mutation():
    """encounter_summary_ is snapshotted at fit; mutating the original
    encounter_df afterward must not alter the frozen summary."""
    enc = _ENC_DF.copy()
    gpf = GratefulPatientFeaturizer(encounter_df=enc)
    gpf.fit(_DONOR_X)
    frozen = gpf.encounter_summary_.copy()

    # Mutate the caller's DataFrame in every way possible after fit.
    enc.loc[0, "service_line"] = "oncology"
    enc.loc[:, "discharge_date"] = "1999-01-01"
    enc.drop(index=1, inplace=True)

    pd.testing.assert_frame_equal(gpf.encounter_summary_, frozen)


def test_grateful_patient_transform_is_idempotent():
    """transform twice on the same X yields identical output."""
    gpf = GratefulPatientFeaturizer(encounter_df=_ENC_DF.copy())
    gpf.fit(_DONOR_X)
    out1 = gpf.transform(_DONOR_X)
    out2 = gpf.transform(_DONOR_X)
    assert out1.shape == (3, 4)
    np.testing.assert_array_equal(out1, out2)


# ---------------------------------------------------------------------------
# 2. WealthPercentileTransformer
# ---------------------------------------------------------------------------

def test_wealth_percentile_rank_independent_of_batch_size():
    """The percentile rank of a fixed probe row is computed against the
    frozen fit-time distribution (percentile_lookup_), so it must be
    identical regardless of how many other rows share the transform batch."""
    rng = np.random.default_rng(0)
    train = pd.DataFrame({"net_worth": rng.uniform(0.0, 1e6, 200)})
    wpt = WealthPercentileTransformer(wealth_cols=["net_worth"])
    wpt.fit(train)

    probe = pd.DataFrame({"net_worth": [500_000.0]})
    small = pd.concat([probe, train.iloc[:3]], ignore_index=True)
    big = pd.concat([probe, train], ignore_index=True)

    out_small = wpt.transform(small)
    out_big = wpt.transform(big)

    # Output columns: [net_worth, net_worth_pct_rank]; probe is row 0.
    rank_col = list(wpt.get_feature_names_out()).index("net_worth_pct_rank")
    assert out_small.shape[0] == 4
    assert out_big.shape[0] == 201
    assert out_small[0, rank_col] == out_big[0, rank_col]


# ---------------------------------------------------------------------------
# 3. ShareOfWalletScorer
# ---------------------------------------------------------------------------

def test_share_of_wallet_score_unchanged_by_outliers_in_batch():
    """wealth_scale_ (95th pct) is frozen at fit and used to clip during
    transform. Appending extreme-wealth outliers to a transform batch must
    not change the score of a fixed probe row."""
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(0.0, 1e6, (50, 3))
    scorer = ShareOfWalletScorer(capacity_col_idx=0, epsilon=1.0)
    scorer.fit(Xtr)
    scale_before = scorer.wealth_scale_

    probe = np.array([[300_000.0, 100_000.0, 50_000.0]])
    base = scorer.transform(probe)

    outliers = np.array([[1e12, 1e12, 1e12], [9e11, 8e11, 7e11]])
    with_outliers = scorer.transform(np.vstack([probe, outliers]))

    assert base[0, 0] == with_outliers[0, 0]
    # Frozen fit statistic untouched by transform.
    assert scorer.wealth_scale_ == scale_before


# ---------------------------------------------------------------------------
# 4. WealthScreeningImputerKNN (strategy="knn")
# ---------------------------------------------------------------------------

def test_knn_imputation_is_batch_independent_for_fixed_row():
    """KNNImputer finds neighbours in the FROZEN fit data, not the transform
    batch, so a fixed probe row imputes identically whether transformed alone
    or batched with other (missing-heavy) rows."""
    rng = np.random.default_rng(42)
    Xtr = rng.uniform(0.0, 1e6, (60, 3))
    Xtr[rng.random((60, 3)) < 0.3] = np.nan

    imp = WealthScreeningImputerKNN(
        strategy="knn", n_neighbors=3, add_indicator=True
    )
    imp.fit(Xtr)
    # Frozen fitted attributes exist after fit.
    assert imp.knn_imputer_ is not None
    assert hasattr(imp, "imputed_cols_")

    probe = np.array([[np.nan, 500_000.0, np.nan]])
    others = rng.uniform(0.0, 1e6, (25, 3))
    others[rng.random((25, 3)) < 0.5] = np.nan

    alone = imp.transform(probe)
    batched = imp.transform(np.vstack([probe, others]))

    assert not np.isnan(alone).any()
    np.testing.assert_allclose(alone[0], batched[0])


def test_knn_fit_transform_reproducible_on_training_set():
    """Fitting on a fixed input and transforming the training set is
    deterministic across independent fits (no hidden per-call randomness)."""
    rng = np.random.default_rng(7)
    Xtr = rng.uniform(0.0, 1e6, (40, 3))
    Xtr[rng.random((40, 3)) < 0.25] = np.nan

    imp_a = WealthScreeningImputerKNN(strategy="knn", n_neighbors=5)
    imp_b = WealthScreeningImputerKNN(strategy="knn", n_neighbors=5)
    out_a = imp_a.fit(Xtr).transform(Xtr)
    out_b = imp_b.fit(Xtr).transform(Xtr)

    np.testing.assert_array_equal(out_a, out_b)


# ---------------------------------------------------------------------------
# 5. EncounterRecencyTransformer
# ---------------------------------------------------------------------------

def test_encounter_recency_reference_date_frozen_at_fit():
    """reference_date_ is resolved once in fit. A transform batch holding a
    later encounter must not shift the recency of a fixed probe row."""
    train = pd.DataFrame({
        "last_encounter_date": ["2022-01-01", "2022-06-01", "2023-01-01"]
    })
    t = EncounterRecencyTransformer()
    t.fit(train)
    frozen = t.reference_date_
    assert frozen == pd.Timestamp("2023-01-01")

    probe = pd.DataFrame({"last_encounter_date": ["2022-01-01"]})
    future = pd.DataFrame({
        "last_encounter_date": ["2022-01-01", "2030-12-31", "2029-01-01"]
    })

    alone = t.transform(probe)
    batched = t.transform(future)

    np.testing.assert_allclose(alone[0], batched[0])
    assert t.reference_date_ == frozen  # transform never rewrites fit state


def test_encounter_recency_transform_is_idempotent():
    train = pd.DataFrame({
        "last_encounter_date": ["2021-03-01", "2022-08-15", "2023-02-02"]
    })
    t = EncounterRecencyTransformer(reference_date="2023-06-01").fit(train)
    np.testing.assert_array_equal(t.transform(train), t.transform(train))


# ---------------------------------------------------------------------------
# 6. MatchingGiftFeaturizer
# ---------------------------------------------------------------------------

def test_matching_gift_ratios_frozen_against_param_mutation():
    """match_ratios_ is normalised and snapshotted at fit. Mutating the dict
    the caller passed in afterward must not change transform output."""
    ratios = {"Acme Corp": 1.0}
    t = MatchingGiftFeaturizer(match_ratios=ratios)
    train = pd.DataFrame({
        "employer": ["Acme Corp", "Beta LLC"],
        "gift_amount": [100.0, 200.0],
    })
    t.fit(train)
    before = t.transform(train)

    ratios["Acme Corp"] = 99.0        # caller mutates their own dict
    ratios["Beta LLC"] = 3.0          # ...and adds a new employer

    np.testing.assert_array_equal(t.transform(train), before)


def test_matching_gift_row_output_independent_of_batch():
    """Every feature is row-local; a probe row scores identically alone and
    batched with employers that are absent from the frozen lookup."""
    t = MatchingGiftFeaturizer(match_ratios={"acme corp": 0.5}).fit(
        pd.DataFrame({"employer": ["Acme Corp"], "gift_amount": [100.0]})
    )
    probe = pd.DataFrame({"employer": ["Acme Corp"], "gift_amount": [100.0]})
    batched = pd.DataFrame({
        "employer": ["Acme Corp", "Unknown Inc", ""],
        "gift_amount": [100.0, 5_000_000.0, 1.0],
    })
    np.testing.assert_array_equal(t.transform(probe)[0], t.transform(batched)[0])
