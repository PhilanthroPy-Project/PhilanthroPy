import os

base = "philanthropy/ingest"


def crlf(s: str) -> bytes:
    return s.replace("\n", "\r\n").encode("utf-8")


def patch(filename: str, old: str, new: str) -> None:
    path = os.path.join(base, filename)
    data = open(path, "rb").read()
    old_b = crlf(old)
    new_b = crlf(new)
    assert old_b in data, f"anchor not found in {filename}"
    data = data.replace(old_b, new_b, 1)
    open(path, "wb").write(data)
    print("patched", filename)


# --------------------------------------------------------------------------- #
# Bloomerang
# --------------------------------------------------------------------------- #
bloomerang_old = """    df = _normalise_headers(_to_frame(transactions), _canonical_bloomerang)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Bloomerang transaction export is missing {missing}. Export the "
            f"'Account Number', 'Date' and 'Amount' fields (REST API: "
            f"AccountId, Date, Amount); got {sorted(df.columns)}."
        )

    if exclude_entry_types:
        if "gift_type" in df.columns:
            unwanted = {_entry_type_key(t) for t in exclude_entry_types}
            keys = df["gift_type"].map(_entry_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Bloomerang transaction export has no entry type column, so "
                f"exclude_entry_types={tuple(exclude_entry_types)!r} could not "
                f"be applied: pledges and recurring-donation schedules (if "
                f"any) are summed alongside the payments made against them, "
                f"which counts every committed dollar twice. Add the "
                f"'Transaction Type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )
"""

bloomerang_new = '''    df = _bloomerang_received_rows(transactions, exclude_entry_types)
    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def _bloomerang_received_rows(
    transactions: Union[Iterable[Mapping], pd.DataFrame],
    exclude_entry_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_ENTRY_TYPES,
) -> pd.DataFrame:
    """Normalise a Bloomerang transaction export and keep only the rows
    recording money received, the rows
    :func:`bloomerang_transactions_to_features` counts.

    Shared with the CLI's raw gift log so both count the same gifts.
    """
    df = _normalise_headers(_to_frame(transactions), _canonical_bloomerang)
    if df.empty:
        return df

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Bloomerang transaction export is missing {missing}. Export the "
            f"'Account Number', 'Date' and 'Amount' fields (REST API: "
            f"AccountId, Date, Amount); got {sorted(df.columns)}."
        )

    if exclude_entry_types:
        if "gift_type" in df.columns:
            unwanted = {_entry_type_key(t) for t in exclude_entry_types}
            keys = df["gift_type"].map(_entry_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"Bloomerang transaction export has no entry type column, so "
                f"exclude_entry_types={tuple(exclude_entry_types)!r} could not "
                f"be applied: pledges and recurring-donation schedules (if "
                f"any) are summed alongside the payments made against them, "
                f"which counts every committed dollar twice. Add the "
                f"'Transaction Type' field to the export.",
                stacklevel=2,
            )

    return df
'''

patch("_bloomerang.py", bloomerang_old, bloomerang_new)

# --------------------------------------------------------------------------- #
# DonorPerfect
# --------------------------------------------------------------------------- #
donorperfect_old = """    df = _normalise_headers(_to_frame(gifts), _canonical_donorperfect)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"DonorPerfect gift export is missing {missing}. Export the "
            f"'Donor ID', 'Gift Date' and 'Gift Amount' fields (DB/API: "
            f"DONOR_ID, GIFT_DATE, AMOUNT); got {sorted(df.columns)}."
        )

    if exclude_record_types:
        if "gift_type" in df.columns:
            unwanted = {_record_type_key(t) for t in exclude_record_types}
            keys = df["gift_type"].map(_record_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"DonorPerfect gift export has no record type column, so "
                f"exclude_record_types={tuple(exclude_record_types)!r} could "
                f"not be applied: pledges and split-gift Main totals (if "
                f"any) are summed alongside the payments/splits made "
                f"against them, which counts every committed or split "
                f"dollar twice. Add the 'Record Type' field to the export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )
"""

donorperfect_new = '''    df = _donorperfect_received_rows(gifts, exclude_record_types)
    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def _donorperfect_received_rows(
    gifts: Union[Iterable[Mapping], pd.DataFrame],
    exclude_record_types: Optional[Sequence[str]] = DEFAULT_EXCLUDED_RECORD_TYPES,
) -> pd.DataFrame:
    """Normalise a DonorPerfect gift export and keep only the rows recording
    money received, the rows :func:`donorperfect_gifts_to_features` counts.

    Shared with the CLI's raw gift log so both count the same gifts.
    """
    df = _normalise_headers(_to_frame(gifts), _canonical_donorperfect)
    if df.empty:
        return df

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"DonorPerfect gift export is missing {missing}. Export the "
            f"'Donor ID', 'Gift Date' and 'Gift Amount' fields (DB/API: "
            f"DONOR_ID, GIFT_DATE, AMOUNT); got {sorted(df.columns)}."
        )

    if exclude_record_types:
        if "gift_type" in df.columns:
            unwanted = {_record_type_key(t) for t in exclude_record_types}
            keys = df["gift_type"].map(_record_type_key)
            df = df[~keys.isin(unwanted)]
        else:
            warnings.warn(
                f"DonorPerfect gift export has no record type column, so "
                f"exclude_record_types={tuple(exclude_record_types)!r} could "
                f"not be applied: pledges and split-gift Main totals (if "
                f"any) are summed alongside the payments/splits made "
                f"against them, which counts every committed or split "
                f"dollar twice. Add the 'Record Type' field to the export.",
                stacklevel=2,
            )

    return df
'''

patch("_donorperfect.py", donorperfect_old, donorperfect_new)

# --------------------------------------------------------------------------- #
# Blackbaud CRM
# --------------------------------------------------------------------------- #
blackbaud_old = """    df = _normalise_headers(_to_frame(revenue), _canonical_blackbaud_crm)
    if df.empty:
        return _empty_feature_frame()

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Blackbaud CRM revenue export is missing {missing}. Export the "
            f"'Constituent ID', 'Date' and 'Amount' fields (database columns: "
            f"CONSTITUENTID, DATE, TRANSACTIONAMOUNT); got {sorted(df.columns)}."
        )

    if include_revenue_types:
        if "gift_type" in df.columns:
            wanted = {_revenue_type_key(t) for t in include_revenue_types}
            keys = df["gift_type"].map(_revenue_type_key)
            df = df[(keys == "") | keys.isin(wanted)]
        else:
            warnings.warn(
                f"Blackbaud CRM revenue export has no revenue type "
                f"column, so include_revenue_types="
                f"{tuple(include_revenue_types)!r} could not be applied: "
                f"a pledge or a recurring gift schedule (if any) is summed "
                f"alongside the payments made against it, which counts every "
                f"committed dollar twice. Add the 'Revenue Type' field to the "
                f"export.",
                stacklevel=2,
            )

    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )
"""

blackbaud_new = '''    df = _blackbaud_crm_received_rows(revenue, include_revenue_types)
    if df.empty:
        return _empty_feature_frame()
    return civicrm_contributions_to_features(
        df, reference_date=reference_date, statuses=None
    )


def _blackbaud_crm_received_rows(
    revenue: Union[Iterable[Mapping], pd.DataFrame],
    include_revenue_types: Optional[Sequence[str]] = DEFAULT_INCLUDED_REVENUE_TYPES,
) -> pd.DataFrame:
    """Normalise a Blackbaud CRM revenue export and keep only the rows
    recording money received, the rows
    :func:`blackbaud_crm_revenue_to_features` counts.

    Shared with the CLI's raw gift log so both count the same gifts.
    """
    df = _normalise_headers(_to_frame(revenue), _canonical_blackbaud_crm)
    if df.empty:
        return df

    missing = [col for col in _REQUIRED if col not in df.columns]
    if missing:
        raise KeyError(
            f"Blackbaud CRM revenue export is missing {missing}. Export the "
            f"'Constituent ID', 'Date' and 'Amount' fields (database columns: "
            f"CONSTITUENTID, DATE, TRANSACTIONAMOUNT); got {sorted(df.columns)}."
        )

    if include_revenue_types:
        if "gift_type" in df.columns:
            wanted = {_revenue_type_key(t) for t in include_revenue_types}
            keys = df["gift_type"].map(_revenue_type_key)
            df = df[(keys == "") | keys.isin(wanted)]
        else:
            warnings.warn(
                f"Blackbaud CRM revenue export has no revenue type "
                f"column, so include_revenue_types="
                f"{tuple(include_revenue_types)!r} could not be applied: "
                f"a pledge or a recurring gift schedule (if any) is summed "
                f"alongside the payments made against it, which counts every "
                f"committed dollar twice. Add the 'Revenue Type' field to the "
                f"export.",
                stacklevel=2,
            )

    return df
'''

patch("_blackbaud_crm.py", blackbaud_old, blackbaud_new)
print("all done")
