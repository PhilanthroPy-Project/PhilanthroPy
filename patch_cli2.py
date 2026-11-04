import os

path = "philanthropy/cli.py"
data = open(path, "rb").read()


def crlf(s: str) -> bytes:
    return s.replace("\n", "\r\n").encode("utf-8")


old = """    from . import ingest
    from .ingest._blackbaud_crm import _blackbaud_crm_received_rows
    from .ingest._bloomerang import _bloomerang_received_rows
    from .ingest._civicrm import _civicrm_received_rows
    from .ingest._donorperfect import _donorperfect_received_rows
    from .ingest._nonprofit_cloud import _nonprofit_cloud_received_rows
    from .ingest._npsp import _npsp_received_rows
    from .ingest._raisers_edge import _raisers_edge_received_rows
    from .utils._validation import ensure_local_path

    ensure_local_path(path, "data")
    try:
        if source == "raisers_edge":
            raw = _raisers_edge_received_rows(ingest.read_raisers_edge_gifts(path))
        elif source == "npsp":
            raw = _npsp_received_rows(ingest.read_npsp_opportunities(path))
        elif source == "nonprofit_cloud":
            raw = _nonprofit_cloud_received_rows(ingest.read_nonprofit_cloud_gifts(path))
        elif source == "bloomerang":
            raw = _bloomerang_received_rows(ingest.read_bloomerang_transactions(path))
        elif source == "donorperfect":
            raw = _donorperfect_received_rows(ingest.read_donorperfect_gifts(path))
        elif source == "blackbaud_crm":
            raw = _blackbaud_crm_received_rows(ingest.read_blackbaud_crm_revenue(path))
        else:
            raw = ingest.read_civicrm_contributions(path)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {path}")
    except KeyError as exc:
        # Same as `features`: the bridge names the missing export fields.
        raise SystemExit(str(exc.args[0] if exc.args else exc))
"""

new = """    from .ingest._civicrm import _civicrm_received_rows
    from .utils._validation import ensure_local_path

    ensure_local_path(path, "data")
    readers = _raw_gift_readers()
    try:
        read = readers[source]
    except KeyError:
        raise SystemExit(
            f"Unknown gift source {source!r}; expected one of {tuple(readers)}."
        ) from None
    try:
        raw = read(path)
    except FileNotFoundError:
        raise SystemExit(f"Data file not found: {path}")
    except KeyError as exc:
        # Same as `features`: the bridge names the missing export fields.
        raise SystemExit(str(exc.args[0] if exc.args else exc))
"""

assert crlf(old) in data, "read_raw_gifts body anchor not found"
data = data.replace(crlf(old), crlf(new), 1)

# Insert the registry helper before _read_raw_gifts
anchor = """def _read_raw_gifts(source: str, path: str) -> pd.DataFrame:
"""
helper = (
    '''def _raw_gift_readers() -> Dict[str, Callable[[str], pd.DataFrame]]:
    """The reader-and-filter pair for every ``ingest.GIFT_SOURCES`` name.

    Kept as one registry so adding a source to ``GIFT_SOURCES`` without a
    ``_read_raw_gifts`` branch is caught by
    ``tests/test_cli.py::test_raw_gift_readers_cover_every_gift_source``
    instead of silently reading the export as CiviCRM.
    """
    from . import ingest
    from .ingest._blackbaud_crm import _blackbaud_crm_received_rows
    from .ingest._bloomerang import _bloomerang_received_rows
    from .ingest._donorperfect import _donorperfect_received_rows
    from .ingest._nonprofit_cloud import _nonprofit_cloud_received_rows
    from .ingest._npsp import _npsp_received_rows
    from .ingest._raisers_edge import _raisers_edge_received_rows

    return {
        "civicrm": lambda path: ingest.read_civicrm_contributions(path),
        "raisers_edge": lambda path: _raisers_edge_received_rows(
            ingest.read_raisers_edge_gifts(path)
        ),
        "npsp": lambda path: _npsp_received_rows(ingest.read_npsp_opportunities(path)),
        "nonprofit_cloud": lambda path: _nonprofit_cloud_received_rows(
            ingest.read_nonprofit_cloud_gifts(path)
        ),
        "bloomerang": lambda path: _bloomerang_received_rows(
            ingest.read_bloomerang_transactions(path)
        ),
        "donorperfect": lambda path: _donorperfect_received_rows(
            ingest.read_donorperfect_gifts(path)
        ),
        "blackbaud_crm": lambda path: _blackbaud_crm_received_rows(
            ingest.read_blackbaud_crm_revenue(path)
        ),
    }


'''
    + anchor
)

assert crlf(anchor) in data, "function anchor not found"
data = data.replace(crlf(anchor), crlf(helper), 1)

open(path, "wb").write(data)
print("cli.py registry refactor done")
