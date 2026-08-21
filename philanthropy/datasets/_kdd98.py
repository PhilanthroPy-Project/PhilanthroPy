"""
philanthropy.datasets._kdd98
=============================
Fetcher for the KDD Cup 1998 direct-mail donor dataset.
"""

from __future__ import annotations

import hashlib
import os
import zipfile
from typing import Optional
from urllib.request import urlopen

import pandas as pd

_LEARNING_URL = "https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98lrn.zip"
_LEARNING_MEMBER = "cup98LRN.txt"
# Computed from the file served at _LEARNING_URL; detects a corrupted
# download or a silently changed upstream file, not a cryptographic guarantee.
_LEARNING_SHA256 = "9517071741c689cf9a27aad4a84d453dc9675d2b2981f90d16a776457daf15bf"


def _data_home(data_home: Optional[str]) -> str:
    if data_home is None:
        data_home = os.environ.get(
            "PHILANTHROPY_DATA", os.path.join("~", "philanthropy_data")
        )
    data_home = os.path.expanduser(data_home)
    os.makedirs(data_home, exist_ok=True)
    return data_home


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, dest: str, expected_sha256: str) -> None:
    tmp = dest + ".part"
    with urlopen(url) as response, open(tmp, "wb") as fh:
        for chunk in iter(lambda: response.read(1 << 20), b""):
            fh.write(chunk)

    digest = _sha256(tmp)
    if digest != expected_sha256:
        os.remove(tmp)
        raise OSError(
            f"Downloaded file from {url} does not match the expected checksum "
            f"(expected {expected_sha256}, got {digest}). The download may have "
            "been interrupted, or the file served at that URL has changed."
        )
    os.replace(tmp, dest)


def fetch_kdd98_donors(
    *, data_home: Optional[str] = None, download_if_missing: bool = True
) -> pd.DataFrame:
    """Fetch the KDD Cup 1998 direct-mail donor learning set.

    A real donor-level dataset: 95,412 individuals who gave at least once
    between June 1995 and June 1996, one row per donor, with a full
    per-promotion mail and response history (``ADATE_2``..``ADATE_24``,
    ``RDATE_2``..``RDATE_24``, ``RAMNT_2``..``RAMNT_24``) plus the outcome of
    the 1997 mailing being predicted (``TARGET_B``, ``TARGET_D``). That date
    history is what makes as-of feature construction testable on it rather
    than only on synthetic data; see :func:`generate_synthetic_donor_data` for
    the synthetic equivalent used elsewhere in this library. Most date columns
    are encoded ``YYMM`` (year, month) rather than as a `datetime` dtype.

    This is a **read-only public research dataset**, not your data. Nothing
    about your own donors, gifts, or environment is ever sent anywhere; the
    only network traffic this function makes is fetching the dataset file
    itself, once, to a local cache. It is never called automatically: no
    other function in this library imports it or calls it during `fit` or
    `transform`.

    Under the dataset's terms of use, teaching or training material that uses
    it must not name the sponsoring organisation; cite it only as "KDD Cup
    1998". This docstring follows that condition, and so should anything you
    write based on it.

    Parameters
    ----------
    data_home : str, default=None
        Directory to cache the downloaded archive in. Defaults to the
        ``PHILANTHROPY_DATA`` environment variable if set, else
        ``~/philanthropy_data``.

    download_if_missing : bool, default=True
        If the archive is not already cached, download it. If False and the
        archive is not cached, raise ``OSError`` instead of reaching for the
        network.

    Returns
    -------
    pandas.DataFrame of shape (95412, 481)
        One row per donor, columns as documented in the data dictionary
        below. Column dtypes are pandas' own inference over the raw CSV;
        this function does not recode or impute any of them.

    Raises
    ------
    OSError
        If the archive is not cached and `download_if_missing` is False, or
        if a downloaded archive fails its checksum check.

    Notes
    -----
    Source: the UCI KDD Archive,
    https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html. Field-by-field
    documentation: ``cup98dic.txt`` at the same location. Distributed for
    general research and educational use under the terms stated on that page,
    including the sponsor-naming restriction noted above and a request to
    notify the dataset's contacts of any published results.

    The archive is not vendored with this package (its terms require an
    unmodified, individually-fetched copy); this function downloads it to a
    local cache on first use, the way ``sklearn.datasets.fetch_*`` functions
    do, and every later call reads the cached copy.
    """
    cache_dir = _data_home(data_home)
    archive_path = os.path.join(cache_dir, "cup98lrn.zip")

    if not os.path.exists(archive_path):
        if not download_if_missing:
            raise OSError(
                f"{archive_path} is not cached and download_if_missing=False. "
                "Call with download_if_missing=True to fetch it."
            )
        _download(_LEARNING_URL, archive_path, _LEARNING_SHA256)

    with zipfile.ZipFile(archive_path) as archive:
        with archive.open(_LEARNING_MEMBER) as fh:
            return pd.read_csv(fh, low_memory=False)
