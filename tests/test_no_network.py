"""The package must never send your data anywhere, and must never download
anything behind your back.

README.md, SECURITY.md and docs/explanation/security_review_answers.md all
promise this, and it is the first question an institutional security review
asks, so it is enforced here rather than merely documented.

Two separate guarantees are enforced, because they fail in different ways:

1. **Runtime.** ``no_network`` poisons every socket entry point, then runs a
   full train/score cycle plus a CRM ingest. A telemetry hook or a lazily
   downloaded asset on any of those paths fails immediately.
2. **Import surface.** ``test_no_module_imports_a_network_client`` parses every
   module in the package and fails if one imports a network-capable library
   without being on ``_NETWORK_ALLOWED``. This is the guarantee the runtime
   fixture cannot give: the fixture only covers the code paths its own tests
   walk, so a downloader added to a module no test imports would slip through
   and quietly falsify the README.

``_NETWORK_ALLOWED`` is empty today, because nothing in this package downloads
anything. It exists so that adding a public dataset fetcher later is a
deliberate, reviewed, one-line act with a matching docs change, rather than an
accident nobody notices.
"""

import ast
import pathlib
import socket

import numpy as np
import pandas as pd
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


def test_encounter_path_rejects_remote_scheme(no_network):
    """User-supplied paths are validated before any pandas read: remote
    schemes must raise locally instead of reaching for the network."""
    from philanthropy.preprocessing import (
        EncounterTransformer,
        GratefulPatientFeaturizer,
    )

    donor_frame = pd.DataFrame({"a": [1.0]})
    for path in ("https://example.com/encounters.parquet", "s3://bucket/e.parquet"):
        with pytest.raises(ValueError, match="must be a local file path"):
            GratefulPatientFeaturizer(encounter_path=path).fit(donor_frame)

        with pytest.raises(ValueError, match="must be a local file path"):
            EncounterTransformer(encounter_path=path).fit(donor_frame)


def test_cli_data_path_rejects_remote_scheme(no_network):
    from philanthropy.cli import _read_csv

    with pytest.raises(ValueError, match="must be a local file path"):
        _read_csv("https://example.com/gifts.csv")

    with pytest.raises(ValueError, match="must be a local file path"):
        _read_csv("gs://bucket/prospects.csv")


def test_local_paths_still_load_under_the_guard(no_network, tmp_path):
    from philanthropy.cli import _read_csv

    csv = tmp_path / "gifts.csv"
    csv.write_text("a,b\n1,2\n")
    df = _read_csv(str(csv))
    assert list(df.columns) == ["a", "b"]


# Importing any of these gives a module the ability to open a socket. Note that
# `urllib.parse` is absent on purpose: it is pure string manipulation, and it is
# what `philanthropy.utils._validation.ensure_local_path` uses to *reject*
# remote paths.
_NETWORK_MODULES = frozenset(
    {
        "aiohttp",
        "ftplib",
        "http.client",
        "httpx",
        "requests",
        "smtplib",
        "socket",
        "socketserver",
        "ssl",
        "telnetlib",
        "urllib.request",
        "urllib3",
        "xmlrpc.client",
    }
)

# Package-relative paths permitted to import from _NETWORK_MODULES. Every entry
# must be an explicitly documented, opt-in dataset fetcher that the user calls
# on purpose, and must never run at import time or inside fit/transform.
#
# EMPTY ON PURPOSE. Adding an entry is a policy change, not just a code change:
# update the "does the software send data anywhere?" answer in
# docs/explanation/security_review_answers.md, plus README.md and SECURITY.md,
# in the same pull request, or those three pages start lying.
_NETWORK_ALLOWED = frozenset()


def _imported_modules(path):
    """Yield every module name `path` imports, as dotted strings."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_no_module_imports_a_network_client():
    """No module may import a network client unless it is allowlisted.

    The runtime fixture only covers the paths its own tests walk. This covers
    every module in the package, including ones no test imports, which is where
    a lazily added downloader would otherwise hide.
    """
    package_root = pathlib.Path(__file__).resolve().parent.parent / "philanthropy"
    assert package_root.is_dir(), package_root

    offenders = {}
    for module_path in sorted(package_root.rglob("*.py")):
        relative = module_path.relative_to(package_root.parent).as_posix()
        if relative in _NETWORK_ALLOWED:
            continue
        hits = sorted(set(_imported_modules(module_path)) & _NETWORK_MODULES)
        if hits:
            offenders[relative] = hits

    assert not offenders, (
        "These modules import a network client but are not in _NETWORK_ALLOWED:\n"
        + "\n".join(f"  {name}: {', '.join(mods)}" for name, mods in offenders.items())
        + "\n\nIf this is a deliberate, opt-in, user-invoked dataset fetcher, add "
        "its path to _NETWORK_ALLOWED and update the no-network wording in "
        "README.md, SECURITY.md and docs/explanation/security_review_answers.md "
        "in the same pull request."
    )


def test_the_allowlist_only_names_modules_that_exist():
    """A stale allowlist entry would hide a typo, or permit nothing at all."""
    package_parent = pathlib.Path(__file__).resolve().parent.parent
    missing = [n for n in _NETWORK_ALLOWED if not (package_parent / n).is_file()]
    assert not missing, f"_NETWORK_ALLOWED names files that do not exist: {missing}"
