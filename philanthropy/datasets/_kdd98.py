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

# KDD Cup 1998's own held-out validation file plus its answer key, released
# after the original competition. Same page, same terms as the learning file
# (see the module docstring below); never touched during model fit or the
# learning file's own 55/15/30 split, so it is the file to score a final,
# genuinely-out-of-sample check against.
_VAL_URL = "https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/cup98val.zip"
_VAL_MEMBER = "cup98VAL.txt"
_VAL_SHA256 = "3112e81cc1d830a534078e65b93067805b199848a35fd41db32f1a4286dbab96"

_VALTARGT_URL = "https://kdd.ics.uci.edu/databases/kddcup98/epsilon_mirror/valtargt.txt"
_VALTARGT_SHA256 = "6d2051306c319fa0771bbc3758ad4fde3f521dee58898927629d2a7dc9b09c4c"


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


def fetch_kdd98_val_donors(
    *, data_home: Optional[str] = None, download_if_missing: bool = True
) -> pd.DataFrame:
    """Fetch KDD Cup 1998's own held-out VALIDATION set, merged with its answer key.

    ``fetch_kdd98_donors`` returns the *learning* file, the one every model in
    this library fits and splits (55/15/30, see
    ``scripts/benchmark_models_vs_baselines.py``). This function returns the
    competition's separate *validation* file (``cup98VAL.txt``): 96,367 more
    donors that were never part of the learning file and never touched
    during a learning-file split of any kind. ``TARGET_B``/``TARGET_D``
    (whether and how much each of them gave to the 97NK mailing) were
    withheld from participants during the original competition and released
    afterwards in a separate answer key, ``valtargt.txt``; this function
    downloads both and joins them on ``CONTROLN`` so the result has the same
    shape as :func:`fetch_kdd98_donors`, including ``TARGET_B``/``TARGET_D``.

    Use this only to score a model already fit elsewhere (e.g. on the
    learning file), never to fit one: a validation file used for fitting
    stops being held out.

    This is a **read-only public research dataset**, not your data, under the
    same terms and the same sponsor-naming restriction as
    :func:`fetch_kdd98_donors`; see that function's docstring.

    Parameters
    ----------
    data_home : str, default=None
        Directory to cache the downloaded archives in. Defaults to the
        ``PHILANTHROPY_DATA`` environment variable if set, else
        ``~/philanthropy_data``.

    download_if_missing : bool, default=True
        If either archive is not already cached, download it. If False and
        either is missing, raise ``OSError`` instead of reaching for the
        network.

    Returns
    -------
    pandas.DataFrame of shape (96367, 481)
        One row per donor, the same columns as :func:`fetch_kdd98_donors`.

    Raises
    ------
    OSError
        If an archive is not cached and `download_if_missing` is False, or
        if a downloaded archive fails its checksum check.

    Notes
    -----
    Source: the UCI KDD Archive,
    https://kdd.ics.uci.edu/databases/kddcup98/kddcup98.html, the same page
    and terms as the learning file. ``valtargt.readme`` at the same location
    documents the answer-key file's three columns (``CONTROLN``,
    ``TARGET_B``, ``TARGET_D``).
    """
    cache_dir = _data_home(data_home)
    val_path = os.path.join(cache_dir, "cup98val.zip")
    targt_path = os.path.join(cache_dir, "valtargt.txt")

    if not os.path.exists(val_path):
        if not download_if_missing:
            raise OSError(
                f"{val_path} is not cached and download_if_missing=False. "
                "Call with download_if_missing=True to fetch it."
            )
        _download(_VAL_URL, val_path, _VAL_SHA256)

    if not os.path.exists(targt_path):
        if not download_if_missing:
            raise OSError(
                f"{targt_path} is not cached and download_if_missing=False. "
                "Call with download_if_missing=True to fetch it."
            )
        _download(_VALTARGT_URL, targt_path, _VALTARGT_SHA256)

    with zipfile.ZipFile(val_path) as archive:
        with archive.open(_VAL_MEMBER) as fh:
            donors = pd.read_csv(fh, low_memory=False)
    targets = pd.read_csv(targt_path)
    return donors.merge(targets, on="CONTROLN", how="inner")
