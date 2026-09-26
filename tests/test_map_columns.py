"""
tests/test_map_columns.py
Tests for philanthropy.ingest.map_columns.
"""

import pandas as pd
import pytest

from philanthropy.ingest import map_columns


def test_renames_mapped_columns_and_keeps_unmapped():
    df = pd.DataFrame({"CnID": [1, 2], "Gift Date": ["2025-01-01", "2025-02-01"], "Note": ["a", "b"]})

    out = map_columns(df, {"CnID": "contact_id", "Gift Date": "activity_date"})

    assert list(out.columns) == ["contact_id", "activity_date", "Note"]


def test_ignores_mapping_keys_absent_from_df():
    df = pd.DataFrame({"CnID": [1]})

    out = map_columns(df, {"CnID": "contact_id", "Hours": "hours"})

    assert list(out.columns) == ["contact_id"]


def test_passes_when_all_required_present_after_mapping():
    df = pd.DataFrame({"CnID": [1], "Gift Date": ["2025-01-01"]})

    out = map_columns(
        df,
        {"CnID": "contact_id", "Gift Date": "activity_date"},
        required=["contact_id", "activity_date"],
    )

    assert list(out.columns) == ["contact_id", "activity_date"]


def test_raises_one_error_listing_every_missing_required_column():
    df = pd.DataFrame({"CnID": [1]})

    with pytest.raises(ValueError, match="activity_date, amount"):
        map_columns(
            df,
            {"CnID": "contact_id"},
            required=["contact_id", "activity_date", "amount"],
        )


def test_no_required_columns_never_raises():
    df = pd.DataFrame({"CnID": [1]})

    out = map_columns(df, {"CnID": "contact_id"})

    assert list(out.columns) == ["contact_id"]


def test_does_not_mutate_input_dataframe():
    df = pd.DataFrame({"CnID": [1]})

    map_columns(df, {"CnID": "contact_id"})

    assert list(df.columns) == ["CnID"]


def test_raises_on_collision_naming_the_colliding_sources():
    df = pd.DataFrame({"A": [1], "B": [2]})

    with pytest.raises(ValueError, match="A.*B|B.*A"):
        map_columns(df, {"A": "x", "B": "x"}, required=["x"])
