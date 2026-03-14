"""
tests/test_utils.py
"""

import pandas as pd
from philanthropy.utils import make_donor_dataset


def test_make_donor_dataset_shape():
    df = make_donor_dataset(n_donors=10, random_state=99)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_make_donor_dataset_columns():
    df = make_donor_dataset(n_donors=5)
    expected = {"donor_id", "gift_date", "gift_amount", "appeal_code", "is_major_gift"}
    assert expected.issubset(df.columns)


def test_make_donor_dataset_reproducible():
    df1 = make_donor_dataset(n_donors=20, random_state=7)
    df2 = make_donor_dataset(n_donors=20, random_state=7)
    assert df1.equals(df2)


def test_make_donor_dataset_major_gift_flag():
    df = make_donor_dataset(n_donors=100, major_gift_threshold=500.0, random_state=0)
    flagged = df[df["is_major_gift"]]
    assert (flagged["gift_amount"] >= 500.0).all()
