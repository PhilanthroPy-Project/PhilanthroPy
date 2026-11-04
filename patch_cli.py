import os

path = "philanthropy/cli.py"
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


# 1. Docstring mention of the extra bridges
replace(
    """    through the same row-filter helpers: a Raiser's Edge pledge, an NPSP open
    stage, an unpaid Nonprofit Cloud schedule, or a cancelled or test-mode
    CiviCRM contribution is not a gift here either. Renaming the three
    canonical columns to the gift-log names is the only new logic.
""",
    """    through the same row-filter helpers: a Raiser's Edge pledge, an NPSP open
    stage, an unpaid Nonprofit Cloud schedule, a cancelled or test-mode
    CiviCRM contribution, a Bloomerang commitment, a DonorPerfect pledge or
    split, or a Blackbaud CRM pledge or recurring schedule is not a gift here
    either. Renaming the three canonical columns to the gift-log names is the
    only new logic.
""",
    "docstring",
)

# 2. Imports inside _read_raw_gifts
replace(
    """    from . import ingest
    from .ingest._civicrm import _civicrm_received_rows
    from .ingest._nonprofit_cloud import _nonprofit_cloud_received_rows
    from .ingest._npsp import _npsp_received_rows
    from .ingest._raisers_edge import _raisers_edge_received_rows
""",
    """    from . import ingest
    from .ingest._blackbaud_crm import _blackbaud_crm_received_rows
    from .ingest._bloomerang import _bloomerang_received_rows
    from .ingest._civicrm import _civicrm_received_rows
    from .ingest._donorperfect import _donorperfect_received_rows
    from .ingest._nonprofit_cloud import _nonprofit_cloud_received_rows
    from .ingest._npsp import _npsp_received_rows
    from .ingest._raisers_edge import _raisers_edge_received_rows
""",
    "imports",
)

# 3. Add branches for the three new sources
replace(
    """        elif source == "nonprofit_cloud":
            raw = _nonprofit_cloud_received_rows(ingest.read_nonprofit_cloud_gifts(path))
        else:
            raw = ingest.read_civicrm_contributions(path)
""",
    """        elif source == "nonprofit_cloud":
            raw = _nonprofit_cloud_received_rows(ingest.read_nonprofit_cloud_gifts(path))
        elif source == "bloomerang":
            raw = _bloomerang_received_rows(ingest.read_bloomerang_transactions(path))
        elif source == "donorperfect":
            raw = _donorperfect_received_rows(ingest.read_donorperfect_gifts(path))
        elif source == "blackbaud_crm":
            raw = _blackbaud_crm_received_rows(ingest.read_blackbaud_crm_revenue(path))
        else:
            raw = ingest.read_civicrm_contributions(path)
""",
    "branches",
)

# 4. Import GIFT_SOURCES in _build_parser
replace(
    """def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
""",
    """def _build_parser() -> argparse.ArgumentParser:
    from .ingest import GIFT_SOURCES

    parser = argparse.ArgumentParser(
""",
    "gift-sources import",
)

# 5. train --source choices
replace(
    """    train.add_argument("--source", default=None, choices=_FEATURE_SOURCES,
""",
    """    train.add_argument("--source", default=None, choices=GIFT_SOURCES,
""",
    "train choices",
)

# 6. check --source choices
replace(
    """    check.add_argument("--source", required=True, choices=_FEATURE_SOURCES,
""",
    """    check.add_argument("--source", required=True, choices=GIFT_SOURCES,
""",
    "check choices",
)

open(path, "wb").write(data)
print("cli.py updated")
