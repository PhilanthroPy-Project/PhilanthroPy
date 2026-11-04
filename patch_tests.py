import os

path = "tests/test_cli.py"
data = open(path, "rb").read()


def crlf(s: str) -> bytes:
    return s.replace("\n", "\r\n").encode("utf-8")


def replace(old: str, new: str, label: str) -> None:
    global data
    old_b = crlf(old)
    new_b = crlf(new)
    assert old_b in data, f"anchor not found: {label}"
    data = data.replace(old_b, new_b, 1)
    print("patched", label)


# 1. Extend the raw gift export fixtures with the three new sources.
replace(
    """    "civicrm": (
        "Contact ID,Contribution Date,Total Amount,Contribution Status",
        "1,2025-01-10,1000,Cancelled",
        "1,2025-02-10,100,Completed",
    ),
}
""",
    """    "civicrm": (
        "Contact ID,Contribution Date,Total Amount,Contribution Status",
        "1,2025-01-10,1000,Cancelled",
        "1,2025-02-10,100,Completed",
    ),
    "bloomerang": (
        "Account Number,Date,Amount,Transaction Type",
        "1,2025-01-10,1000,Pledge",
        "1,2025-02-10,100,PledgePayment",
    ),
    "donorperfect": (
        "Donor ID,Gift Date,Gift Amount,Record Type",
        "1,2025-01-10,1000,P",
        "1,2025-02-10,100,G",
    ),
    "blackbaud_crm": (
        "Constituent ID,Date,Amount,Revenue Type",
        "1,2025-01-10,1000,Pledge",
        "1,2025-02-10,100,Payment",
    ),
}
""",
    "fixtures",
)

# 2. Add the coverage and unknown-source tests after the missing-fields test.
replace(
    """def test_raw_gift_log_names_missing_fields_instead_of_a_traceback(tmp_path):
    from philanthropy.cli import _read_raw_gifts

    path = tmp_path / "re.csv"
    path.write_text("Constituent ID,Gift Amount\\n1,100\\n")
    with pytest.raises(SystemExit, match="missing"):
        _read_raw_gifts("raisers_edge", str(path))
""",
    """def test_raw_gift_log_names_missing_fields_instead_of_a_traceback(tmp_path):
    from philanthropy.cli import _read_raw_gifts

    path = tmp_path / "re.csv"
    path.write_text("Constituent ID,Gift Amount\\n1,100\\n")
    with pytest.raises(SystemExit, match="missing"):
        _read_raw_gifts("raisers_edge", str(path))


def test_raw_gift_readers_cover_every_gift_source():
    from philanthropy.cli import _raw_gift_readers
    from philanthropy.ingest import GIFT_SOURCES

    assert set(_raw_gift_readers()) == set(GIFT_SOURCES)


def test_read_raw_gifts_rejects_an_unknown_source(tmp_path):
    from philanthropy.cli import _read_raw_gifts

    path = tmp_path / "gifts.csv"
    path.write_text("Constituent ID,Gift Date,Gift Amount\\n1,2025-01-10,100\\n")
    with pytest.raises(SystemExit, match="Unknown gift source"):
        _read_raw_gifts("not_a_source", str(path))
""",
    "tests",
)

open(path, "wb").write(data)
print("tests updated")
