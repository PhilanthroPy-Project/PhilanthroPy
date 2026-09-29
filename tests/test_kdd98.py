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

from philanthropy.datasets import fetch_kdd98_donors, fetch_kdd98_val_donors


def _write_fixture_archive(path, csv_text="TARGET_B,TARGET_D\n0,0\n1,25\n"):
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("cup98LRN.txt", csv_text)


def _write_val_fixture(cache_dir):
    with zipfile.ZipFile(cache_dir / "cup98val.zip", "w") as zf:
        zf.writestr("cup98VAL.txt", "CONTROLN,AGE\n1,62\n2,45\n3,71\n")
    (cache_dir / "valtargt.txt").write_text("CONTROLN,TARGET_B,TARGET_D\n1,0,0\n2,1,15\n3,0,0\n")


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


def test_fetch_kdd98_val_donors_raises_without_download_and_no_cache(tmp_path):
    with pytest.raises(OSError, match="download_if_missing"):
        fetch_kdd98_val_donors(data_home=str(tmp_path), download_if_missing=False)


def test_fetch_kdd98_val_donors_merges_the_answer_key_on_controln(tmp_path):
    _write_val_fixture(tmp_path)

    df = fetch_kdd98_val_donors(data_home=str(tmp_path), download_if_missing=False)

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["CONTROLN", "AGE", "TARGET_B", "TARGET_D"]
    assert df.shape == (3, 4)
    assert df.set_index("CONTROLN").loc[2, "TARGET_D"] == 15


def test_fetch_kdd98_val_donors_docstring_does_not_name_the_sponsor():
    assert "paralyzed veterans" not in fetch_kdd98_val_donors.__doc__.lower()
    assert "pva" not in fetch_kdd98_val_donors.__doc__.lower()


def test_fetch_kdd98_val_donors_downloads_and_verifies_checksums(tmp_path, monkeypatch):
    import hashlib

    import philanthropy.datasets._kdd98 as kdd98

    val_fixture = tmp_path / "val_fixture.zip"
    with zipfile.ZipFile(val_fixture, "w") as zf:
        zf.writestr("cup98VAL.txt", "CONTROLN,AGE\n1,62\n")
    val_payload = val_fixture.read_bytes()
    targt_payload = b"CONTROLN,TARGET_B,TARGET_D\n1,1,10\n"

    monkeypatch.setattr(kdd98, "_VAL_SHA256", hashlib.sha256(val_payload).hexdigest())
    monkeypatch.setattr(kdd98, "_VALTARGT_SHA256", hashlib.sha256(targt_payload).hexdigest())

    payloads = {kdd98._VAL_URL: val_payload, kdd98._VALTARGT_URL: targt_payload}

    class _FakeResponse:
        def __init__(self, payload):
            self._payload = payload

        def __enter__(self):
            return io.BytesIO(self._payload)

        def __exit__(self, *exc_info):
            return False

    monkeypatch.setattr(kdd98, "urlopen", lambda url: _FakeResponse(payloads[url]))

    data_home = tmp_path / "cache"
    df = fetch_kdd98_val_donors(data_home=str(data_home), download_if_missing=True)

    assert list(df.columns) == ["CONTROLN", "AGE", "TARGET_B", "TARGET_D"]
    assert (data_home / "cup98val.zip").exists()
    assert (data_home / "valtargt.txt").exists()
    assert not (data_home / "cup98val.zip.part").exists()
