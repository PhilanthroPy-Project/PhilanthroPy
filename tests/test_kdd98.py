"""
tests/test_kdd98.py
====================
Unit tests for the KDD Cup 1998 donor-data fetcher. None of these tests touch
the network: every path here either supplies a pre-populated cache
(`download_if_missing=False`) or monkeypatches the download itself.
"""

import io
import zipfile

import pandas as pd
import pytest

from philanthropy.datasets import fetch_kdd98_donors


def _write_fixture_archive(path, csv_text="TARGET_B,TARGET_D\n0,0\n1,25\n"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cup98LRN.txt", csv_text)


def test_fetch_kdd98_donors_raises_without_download_and_no_cache(tmp_path):
    with pytest.raises(OSError, match="download_if_missing"):
        fetch_kdd98_donors(data_home=str(tmp_path), download_if_missing=False)


def test_fetch_kdd98_donors_returns_expected_columns(tmp_path):
    _write_fixture_archive(tmp_path / "cup98lrn.zip")

    df = fetch_kdd98_donors(data_home=str(tmp_path), download_if_missing=False)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["TARGET_B", "TARGET_D"]
    assert df.shape == (2, 2)


def test_fetch_kdd98_donors_docstring_does_not_name_the_sponsor():
    # The dataset's terms of use forbid naming the sponsoring charity in
    # teaching or training material; this docstring is exactly that.
    assert "paralyzed veterans" not in fetch_kdd98_donors.__doc__.lower()
    assert "pva" not in fetch_kdd98_donors.__doc__.lower()


def test_data_home_env_var_is_honoured(tmp_path, monkeypatch):
    _write_fixture_archive(tmp_path / "cup98lrn.zip")
    monkeypatch.setenv("PHILANTHROPY_DATA", str(tmp_path))

    df = fetch_kdd98_donors(download_if_missing=False)

    assert list(df.columns) == ["TARGET_B", "TARGET_D"]


def test_download_caches_a_good_file_and_reads_it(tmp_path, monkeypatch):
    import hashlib

    import philanthropy.datasets._kdd98 as kdd98

    fixture = tmp_path / "fixture.zip"
    _write_fixture_archive(fixture)
    payload = fixture.read_bytes()
    monkeypatch.setattr(kdd98, "_LEARNING_SHA256", hashlib.sha256(payload).hexdigest())

    class _FakeResponse:
        def __enter__(self):
            return io.BytesIO(payload)

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(kdd98, "urlopen", lambda url: _FakeResponse())

    data_home = tmp_path / "cache"
    df = fetch_kdd98_donors(data_home=str(data_home), download_if_missing=True)

    assert list(df.columns) == ["TARGET_B", "TARGET_D"]
    assert (data_home / "cup98lrn.zip").exists()
    assert not (data_home / "cup98lrn.zip.part").exists()


def test_download_verifies_checksum_and_discards_a_bad_file(tmp_path, monkeypatch):
    import philanthropy.datasets._kdd98 as kdd98

    monkeypatch.setattr(kdd98, "_LEARNING_SHA256", "0" * 64)

    class _FakeResponse:
        def __enter__(self):
            return io.BytesIO(b"not the real archive")

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(kdd98, "urlopen", lambda url: _FakeResponse())

    with pytest.raises(OSError, match="checksum"):
        fetch_kdd98_donors(data_home=str(tmp_path), download_if_missing=True)

    assert not (tmp_path / "cup98lrn.zip").exists()
    assert not (tmp_path / "cup98lrn.zip.part").exists()
