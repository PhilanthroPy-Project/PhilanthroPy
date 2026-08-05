"""The package must never open a socket.

README.md and docs/explanation/security_review_answers.md both promise that
PhilanthroPy makes no network calls — no telemetry, no license check, no
third-party data append. That promise is the first question an institutional
security review asks, so it is enforced here rather than merely documented.

The guard poisons every socket entry point, then runs a full train/score cycle
plus a CRM ingest. Any HTTP client, telemetry hook, or lazily downloaded asset
added later fails this test instead of shipping.
"""

import socket

import numpy as np
import pytest

from philanthropy.datasets import generate_synthetic_donor_data
from philanthropy.ingest import (
    civicrm_contributions_to_features,
    read_civicrm_contributions,
)
from philanthropy.models import DonorPropensityModel
from philanthropy.preprocessing import WealthScreeningImputer

FEATURES = ["total_gift_amount", "years_active", "event_attendance_count"]


class NetworkAccessAttempted(AssertionError):
    """Raised when library code tries to reach the network."""


@pytest.fixture
def no_network(monkeypatch):
    """Make every socket operation raise."""

    def deny(*args, **kwargs):
        raise NetworkAccessAttempted(
            "PhilanthroPy attempted a network call. The package must stay offline; "
            "see the no-network promise in README.md."
        )

    monkeypatch.setattr(socket, "socket", deny)
    monkeypatch.setattr(socket, "create_connection", deny)
    monkeypatch.setattr(socket, "getaddrinfo", deny)
    monkeypatch.setattr(socket, "gethostbyname", deny)
    return deny


def test_train_and_score_make_no_network_calls(no_network):
    df = generate_synthetic_donor_data(n_samples=200, random_state=0)
    X = df[FEATURES].to_numpy()

    model = DonorPropensityModel(n_estimators=10, random_state=0)
    model.fit(X, df["is_major_donor"].to_numpy())
    scores = model.predict_affinity_score(X)

    assert scores.shape == (200,)


def test_imputer_makes_no_network_calls(no_network):
    X = np.array([[100.0, 2.0], [np.nan, 3.0], [300.0, np.nan]])

    imputer = WealthScreeningImputer()
    assert imputer.fit_transform(X).shape == X.shape


def test_civicrm_ingest_makes_no_network_calls(no_network, tmp_path):
    csv = tmp_path / "contributions.csv"
    csv.write_text(
        "contact_id,receive_date,total_amount,contribution_status\n"
        "1,2024-03-01,500.00,Completed\n"
        "1,2025-03-01,750.00,Completed\n"
        "2,2024-06-15,50.00,Completed\n"
        "2,2024-07-01,999.00,Refunded\n"
    )

    contributions = read_civicrm_contributions(csv)
    features = civicrm_contributions_to_features(contributions)

    assert len(features) == 2
    # The refunded gift is excluded, so the smaller donor's lifetime giving
    # stays at 50 rather than 1049.
    assert sorted(features["total_gift_amount"]) == [50.0, 1250.0]


def test_the_guard_itself_actually_bites(no_network):
    """Guard against a silently ineffective fixture."""
    with pytest.raises(NetworkAccessAttempted):
        socket.socket()
