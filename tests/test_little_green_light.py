import warnings

import pandas as pd
import pytest

from philanthropy.ingest import (
    DEFAULT_EXCLUDED_LGL_GIFT_TYPES,
    little_green_light_gifts_to_features,
    read_little_green_light_gifts,
)
from philanthropy.ingest._civicrm import _FEATURE_DTYPES


@pytest.fixture
def gifts():
    return [
        {
            "LGL Constituent ID": "88",
            "Gift date": "2025-01-10",
            "Amount": "1200.00",
            "Gift type": "Pledge",
            "Fund": "General",
        },
        {
            "LGL Constituent ID": "88",
            "Gift date": "2025-02-10",
            "Amount": "100.00",
            "Gift type": "Gift",
            "Fund": "General",
        },
        {
            "LGL Constituent ID": "88",
            "Gift date": "2025-03-10",
            "Amount": "100.00",
            "Gift type": "Gift",
            "Fund": "Research",
        },
    ]


def test_pledge_is_excluded_but_payments_remain(gifts):
    features = little_green_light_gifts_to_features(gifts)
    assert float(features.loc["88", "total_gift_amount"]) == 200.0
    assert int(features.loc["88", "gift_count"]) == 2


@pytest.mark.parametrize("gift_type", ["pledge", "PLEDGE", " Pledge "])
def test_pledge_matching_ignores_case_and_spacing(gift_type):
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "500", "Gift type": gift_type},
        {"LGL Constituent ID": "1", "Gift date": "2025-01-02",
         "Amount": "25", "Gift type": "Gift"},
    ]
    features = little_green_light_gifts_to_features(rows)
    assert float(features.loc["1", "total_gift_amount"]) == 25.0


@pytest.mark.parametrize("gift_type", ["Gift", "Other Income", "In Kind"])
def test_non_pledge_types_are_kept(gift_type):
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "25", "Gift type": gift_type}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert float(features.loc["1", "total_gift_amount"]) == 25.0


def test_exclusion_can_be_disabled(gifts):
    features = little_green_light_gifts_to_features(
        gifts, exclude_gift_types=None
    )
    assert float(features.loc["88", "total_gift_amount"]) == 1400.0
    assert int(features.loc["88", "gift_count"]) == 3


def test_empty_exclusion_sequence_disables_filter(gifts):
    features = little_green_light_gifts_to_features(
        gifts, exclude_gift_types=()
    )
    assert float(features.loc["88", "total_gift_amount"]) == 1400.0


def test_custom_exclusion_replaces_default(gifts):
    features = little_green_light_gifts_to_features(
        gifts, exclude_gift_types=("Gift",)
    )
    assert float(features.loc["88", "total_gift_amount"]) == 1200.0


def test_default_exclusion_is_documented():
    assert "Pledge" in DEFAULT_EXCLUDED_LGL_GIFT_TYPES


def test_missing_gift_type_warns():
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "10"}
    ]
    with pytest.warns(UserWarning, match="no gift type column"):
        features = little_green_light_gifts_to_features(rows)
    assert float(features.loc["1", "total_gift_amount"]) == 10.0


def test_disabling_filter_does_not_warn():
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "10"}
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        little_green_light_gifts_to_features(
            rows, exclude_gift_types=None
        )


@pytest.mark.parametrize(
    "id_header", ["LGL Constituent ID", "constituent_id", "Contact ID"]
)
def test_constituent_id_aliases(id_header):
    rows = [
        {id_header: "0088", "Gift date": "2025-01-01",
         "Amount": "25", "Gift type": "Gift"}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert "0088" in features.index


@pytest.mark.parametrize("date_header", ["Gift date", "Gift Date"])
def test_gift_date_aliases(date_header):
    rows = [
        {"LGL Constituent ID": "1", date_header: "2025-01-01",
         "Amount": "25", "Gift type": "Gift"}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert features.loc["1", "last_gift_date"] == pd.Timestamp("2025-01-01")


@pytest.mark.parametrize("amount_header", ["Amount", "Gift Amount"])
def test_amount_aliases(amount_header):
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         amount_header: "25", "Gift type": "Gift"}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert float(features.loc["1", "total_gift_amount"]) == 25.0


def test_currency_formatted_amounts():
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "$1,250.00", "Gift type": "Gift"}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert float(features.loc["1", "total_gift_amount"]) == 1250.0


def test_schema_and_dtypes(gifts):
    features = little_green_light_gifts_to_features(gifts)
    assert list(features.columns) == list(_FEATURE_DTYPES)
    assert features.index.name == "contact_id"
    for column, dtype in _FEATURE_DTYPES.items():
        assert str(features[column].dtype) == dtype


def test_fund_counts_distinct_financial_types(gifts):
    features = little_green_light_gifts_to_features(gifts)
    assert int(features.loc["88", "distinct_financial_types"]) == 2


def test_empty_input_returns_typed_frame():
    features = little_green_light_gifts_to_features([])
    assert features.empty
    assert list(features.columns) == list(_FEATURE_DTYPES)


def test_all_pledges_returns_typed_empty_frame():
    rows = [
        {"LGL Constituent ID": "1", "Gift date": "2025-01-01",
         "Amount": "500", "Gift type": "Pledge"}
    ]
    features = little_green_light_gifts_to_features(rows)
    assert features.empty
    assert list(features.columns) == list(_FEATURE_DTYPES)


def test_missing_required_columns():
    rows = [{"LGL Constituent ID": "1", "Gift type": "Gift"}]
    with pytest.raises(KeyError, match="Little Green Light"):
        little_green_light_gifts_to_features(rows)


def test_read_single_csv(tmp_path):
    path = tmp_path / "gifts.csv"
    path.write_text(
        "LGL Constituent ID,Gift date,Amount,Gift type\n"
        "88,2025-01-10,1200,Pledge\n"
        "88,2025-02-10,100,Gift\n"
    )
    raw = read_little_green_light_gifts(path)
    assert len(raw) == 2
    features = little_green_light_gifts_to_features(raw)
    assert float(features.loc["88", "total_gift_amount"]) == 100.0


def test_read_directory(tmp_path):
    (tmp_path / "jan.csv").write_text(
        "LGL Constituent ID,Gift date,Amount,Gift type\n"
        "88,2025-01-10,100,Gift\n"
    )
    (tmp_path / "feb.csv").write_text(
        "LGL Constituent ID,Gift date,Amount,Gift type\n"
        "88,2025-02-10,250,Gift\n"
    )
    features = little_green_light_gifts_to_features(
        read_little_green_light_gifts(tmp_path)
    )
    assert float(features.loc["88", "total_gift_amount"]) == 350.0


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        read_little_green_light_gifts(tmp_path / "missing.csv")