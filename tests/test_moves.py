"""Tests for MovesManagementClassifier.

Moved here from the deleted tests/test_coverage_boost.py, a filename that named
a metric as the goal rather than the behaviour under test.
"""

import numpy as np
import pytest

from philanthropy.models import MovesManagementClassifier

_STAGES = ["IDENTIFY", "QUALIFY", "CULTIVATE"]


@pytest.fixture
def stage_Xy():
    rng = np.random.default_rng(0)
    X = rng.random((30, 5))
    y = np.asarray(_STAGES * 10)
    return X, y


def test_predict_proba_has_one_column_per_stage(stage_Xy):
    X, y = stage_Xy
    clf = MovesManagementClassifier(max_iter=10, random_state=0).fit(X, y)

    proba = clf.predict_proba(X)
    assert proba.shape == (30, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0)
    assert sorted(clf.classes_) == sorted(_STAGES)


def test_action_priority_reports_a_stage_and_confidence_per_donor(stage_Xy):
    X, y = stage_Xy
    clf = MovesManagementClassifier(max_iter=10, random_state=0).fit(X, y)

    priority = clf.action_priority(X)
    assert set(priority) == {"stage", "confidence", "portfolio_summary"}

    assert len(priority["stage"]) == 30
    assert set(priority["stage"]) <= set(_STAGES)
    # The reported stage is argmax of predict_proba and confidence is its value.
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(priority["confidence"], proba.max(axis=1))
    np.testing.assert_array_equal(
        priority["stage"],
        clf.label_encoder_.inverse_transform(proba.argmax(axis=1)),
    )


def test_action_priority_summary_counts_every_donor(stage_Xy):
    X, y = stage_Xy
    clf = MovesManagementClassifier(max_iter=10, random_state=0).fit(X, y)

    summary = clf.action_priority(X)["portfolio_summary"]
    assert sum(summary.values()) == 30
    assert set(summary) <= set(_STAGES)
