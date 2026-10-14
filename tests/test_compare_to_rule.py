"""
tests/test_compare_to_rule.py
"""

import numpy as np
import pandas as pd
import pytest

from philanthropy.datasets import make_donor_panel
from philanthropy.model_selection import compare_to_rule
from philanthropy.utils import LABEL_FLOORS

VERDICTS = {"Beats the rule", "About the same as the rule", "Loses to the rule"}


def _gift_log(n_donors, n_years, gifts_per_year, give_next, seed, amount=100.0):
    """Donors give ``gifts_per_year[d]`` gifts a year (one on June 30, so
    recency is the same for everyone); whether donor ``d`` gives in year
    ``t + 1`` is a coin with probability ``give_next[d]``."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_donors):
        gives = True
        for t in range(n_years):
            if gives:
                for k in range(gifts_per_year[d]):
                    month = 6 if k == 0 else 1 + k
                    rows.append({"donor_id": f"d{d}", "gift_date": f"{2015 + t}-{month:02d}-28",
                                 "gift_amount": amount / gifts_per_year[d]})
            gives = rng.random() < give_next[d]
    return pd.DataFrame(rows)


def test_model_wins_when_the_label_follows_a_feature_no_rule_ranks_by():
    # Lapse is driven by the number of gifts in the year, which none of the
    # three lapse rules (recency, streak, trend) look at.
    n = 1200
    gifts_per_year = np.where(np.arange(n) % 2 == 0, 1, 4)
    give_next = np.where(gifts_per_year == 1, 0.15, 0.95)
    out = compare_to_rule(
        _gift_log(n, 7, gifts_per_year, give_next, seed=0), task="lapse", n_bootstrap=300,
    )
    assert out["label_floor"] == "run"
    assert out["verdict"] == "Beats the rule"
    assert out["method"] == "model"
    assert out["model_value"] > out["rule_value"]
    lo, hi = out["difference_ci"]
    assert 0 < lo <= hi


def test_rule_wins_when_the_label_is_the_rule():
    # Every donor gives the same amount every year, so "last period's total"
    # predicts next year exactly and no fitted model can match zero error.
    rng = np.random.default_rng(0)
    amounts = np.round(rng.lognormal(4.0, 1.0, size=400), 2)
    gifts = pd.DataFrame([
        {"donor_id": f"d{d}", "gift_date": f"{fy}-03-01", "gift_amount": amounts[d]}
        for d in range(len(amounts)) for fy in range(2016, 2022)
    ])
    out = compare_to_rule(gifts, task="ask", n_bootstrap=300)
    assert out["rule"] == "last period's total"
    assert out["rule_value"] == 0.0
    assert out["verdict"] == "Loses to the rule"
    assert out["method"] == "rule"
    assert out["model_value"] > 0


def test_too_few_labels_keeps_the_rule_without_fitting():
    gifts = make_donor_panel(n_donors=60, n_years=6, random_state=0)["gifts"]
    out = compare_to_rule(gifts, task="lapse")
    assert out["verdict"] == "not enough labels"
    assert out["label_floor"] == "not enough labels"
    assert out["method"] == "rule"
    assert out["model_value"] is None and out["difference_ci"] is None
    assert out["rule"] is not None and out["rule_value"] is not None


def test_one_labelled_year_cannot_be_tested():
    gifts = pd.DataFrame({
        "donor_id": ["a", "a", "b", "b"],
        "gift_date": ["2019-08-01", "2020-08-01", "2019-08-01", "2020-08-01"],
        "gift_amount": [10, 20, 30, 40],
    })
    out = compare_to_rule(gifts, task="response")
    assert out["verdict"].startswith("can't test")
    assert out["method"] == "rule"
    assert compare_to_rule(gifts.iloc[:0], task="lapse")["verdict"].startswith("can't test")


def test_partial_current_year_never_labels_the_one_before_it():
    gifts = make_donor_panel(n_donors=400, n_years=5, random_state=0)["gifts"]
    # Cut the file in the middle of FY2022 (July-start fiscal years), so
    # FY2022 is incomplete: FY2021 cannot be labelled from it.
    out = compare_to_rule(gifts, task="lapse", as_of="2021-12-31", n_bootstrap=20)
    assert out["test_fiscal_year"] == 2020
    full = compare_to_rule(gifts, task="lapse", as_of="2022-06-30", n_bootstrap=20)
    assert full["test_fiscal_year"] == 2021


@pytest.mark.parametrize("task", ["upgrade", "lapse", "response", "ask"])
def test_every_task_reports_the_same_keys(task):
    gifts = make_donor_panel(n_donors=800, n_years=5, random_state=2)["gifts"]
    out = compare_to_rule(gifts, task=task, n_bootstrap=20)
    assert set(out) == {
        "task", "method", "verdict", "rule", "n_rules_tried", "test_fiscal_year", "n_train",
        "n_test", "metric", "model_value", "rule_value", "difference_ci", "label_floor",
    }
    assert out["method"] in ("model", "rule")
    assert out["verdict"] in VERDICTS or out["verdict"] == "not enough labels"
    assert out["metric"] == ("mae" if task == "ask" else "top10pct_hit_rate")
    assert out["n_rules_tried"] == (2 if task == "response" else 3)
    if out["verdict"] in VERDICTS:
        assert (out["method"] == "model") == (out["verdict"] == "Beats the rule")
    else:
        assert out["label_floor"] == "not enough labels"
        assert out["n_train"] < 6 * LABEL_FLOORS["major_gift" if task != "lapse" else "lapse"]


def test_bad_task_raises():
    with pytest.raises(ValueError, match="task"):
        compare_to_rule(pd.DataFrame(columns=["donor_id", "gift_date", "gift_amount"]), task="retention")
