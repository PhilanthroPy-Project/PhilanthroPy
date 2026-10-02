"""
philanthropy.datasets._psid
============================
Local-file reader for the PSID Philanthropy Panel Study-style giving and
volunteering history (PSID Data Center individual-level cross-year extract).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

import pandas as pd

# One entry per PSID wave read by this reader, built from a "Current Year
# Heads Individual Data" cross-year extract (PSID Data Center job 365729,
# IND domain, 483 variables, 2001-2023). Every ER/S variable number below
# was read off that job's own .do (variable labels) and cross-checked
# against its codebook HTML, not typed from memory or an external index.
# ER/S numbers are fixed, study-wide PSID identifiers: ER20047 names the
# same variable in any job that selects it. A job with a smaller variable
# selection than 365729 simply lacks some of these columns; a wave whose
# relation_var is entirely absent is skipped (see load_psid_philanthropy),
# while a wave whose relation_var is present but another of its variables
# below is missing raises KeyError rather than silently misreading a
# column, since that combination means this table and the extract disagree
# about what was selected.
_WAVES: List[Dict[str, Any]] = [
    {
        "year": 2001,
        "relation_var": "ER33603",
        "income_var": "ER20456",
        "wealth1_var": "S516",
        "wealth2_var": "S517",
        "itemized_var": "ER19162",
        "volunteer_head_hours_var": "ER20089",
        "volunteer_spouse_hours_var": "ER20097",
        "hours_kind": "annual",
        "giving": {
            "religious": "ER20047", "combo": "ER20053", "needy": "ER20059",
            "health": "ER20065", "education": "ER20071",
            # 2001 only: a single combined amount for youth/cultural/
            # community/environment/international/other, asked as one
            # multi-select checkpoint (T7A-F) rather than per-category
            # questions like every later wave.
            "checkpoint_other_2001": "ER20083",
        },
    },
    {
        "year": 2003, "relation_var": "ER33703", "income_var": "ER24099",
        "wealth1_var": "S616", "wealth2_var": "S617", "itemized_var": "ER22535",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        # Annual hours "regularly" volunteered, asked once per organization
        # type (up to 14 slots: M15I, M18I, M21H, ..., M43H) rather than
        # once per head/spouse like 2001's T8A/T10A or 2017+'s F1E. The
        # codebook's compact label index has no question text or universe
        # note distinguishing head from spouse here, and 2003/2005 have no
        # parallel head-only/spouse-only battery the way 2011 onward do;
        # treat this as one household-level total, not a head or spouse
        # figure, and never merge it with head_volunteer_hours_annual.
        "regular_org_hours_vars": [
            "ER23563", "ER23573", "ER23582", "ER23591", "ER23600", "ER23609",
            "ER23618", "ER23627", "ER23636", "ER23645", "ER23654", "ER23663",
            "ER23673", "ER23683",
        ],
        "giving": {
            "religious": "ER23483", "combo": "ER23489", "needy": "ER23495",
            "health": "ER23501", "education": "ER23507", "youth": "ER23513",
            "cultural": "ER23519", "community": "ER23525",
            "environment": "ER23531", "international": "ER23537",
            "other": "ER23543",
        },
    },
    {
        "year": 2005, "relation_var": "ER33803", "income_var": "ER28037",
        "wealth1_var": "S716", "wealth2_var": "S717", "itemized_var": "ER26516",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "regular_org_hours_vars": [
            "ER27533", "ER27543", "ER27552", "ER27561", "ER27570", "ER27579",
            "ER27588", "ER27597", "ER27606", "ER27615", "ER27624", "ER27633",
            "ER27643", "ER27653",
        ],
        "giving": {
            "religious": "ER27451", "combo": "ER27457", "needy": "ER27463",
            "health": "ER27469", "education": "ER27475", "youth": "ER27481",
            "cultural": "ER27487", "community": "ER27493",
            "environment": "ER27499", "international": "ER27505",
            "other": "ER27511",
        },
    },
    {
        "year": 2007, "relation_var": "ER33903", "income_var": "ER41027",
        "wealth1_var": "S816", "wealth2_var": "S817", "itemized_var": "ER37534",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "giving": {
            "religious": "ER40622", "combo": "ER40628", "needy": "ER40634",
            "health": "ER40640", "education": "ER40646", "youth": "ER40652",
            "cultural": "ER40658", "community": "ER40664",
            "environment": "ER40670", "international": "ER40676",
            "other": "ER40682",
        },
    },
    {
        "year": 2009, "relation_var": "ER34003", "income_var": "ER46935",
        "wealth1_var": "ER46968", "wealth2_var": "ER46970", "itemized_var": "ER43525",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "giving": {
            "religious": "ER46600", "combo": "ER46606", "needy": "ER46612",
            "health": "ER46618", "education": "ER46624", "youth": "ER46630",
            "cultural": "ER46636", "community": "ER46642",
            "environment": "ER46648", "international": "ER46654",
            "other": "ER46660",
        },
    },
    {
        "year": 2011, "relation_var": "ER34103", "income_var": "ER52343",
        "wealth1_var": "ER52392", "wealth2_var": "ER52394", "itemized_var": "ER48850",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "giving": {
            "religious": "ER51961", "combo": "ER51967", "needy": "ER51973",
            "health": "ER51979", "education": "ER51985", "youth": "ER51991",
            "cultural": "ER51997", "community": "ER52003",
            "environment": "ER52009", "international": "ER52015",
            "other": "ER52021",
        },
    },
    {
        "year": 2013, "relation_var": "ER34203", "income_var": "ER58152",
        "wealth1_var": "ER58209", "wealth2_var": "ER58211", "itemized_var": "ER54593",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "giving": {
            "religious": "ER57730", "combo": "ER57736", "needy": "ER57742",
            "health": "ER57748", "education": "ER57754", "youth": "ER57760",
            "cultural": "ER57766", "community": "ER57772",
            "environment": "ER57778", "international": "ER57784",
            "other": "ER57791",
        },
    },
    {
        "year": 2015, "relation_var": "ER34303", "income_var": "ER65349",
        "wealth1_var": "ER65406", "wealth2_var": "ER65408", "itemized_var": "ER61704",
        "volunteer_head_hours_var": None, "volunteer_spouse_hours_var": None,
        "hours_kind": None,
        "giving": {
            "religious": "ER64910", "combo": "ER64916", "needy": "ER64922",
            "health": "ER64928", "education": "ER64934", "youth": "ER64940",
            "cultural": "ER64946", "community": "ER64952",
            "environment": "ER64958", "international": "ER64964",
            "other": "ER64971",
        },
    },
    {
        "year": 2017, "relation_var": "ER34503", "income_var": "ER71426",
        "wealth1_var": "ER71483", "wealth2_var": "ER71485", "itemized_var": "ER67757",
        "volunteer_head_hours_var": "ER66720", "volunteer_spouse_hours_var": "ER66733",
        "hours_kind": "typical_week",
        "giving": {
            "religious": "ER71042", "combo": "ER71044", "needy": "ER71046",
            "health": "ER71048", "education": "ER71050", "youth": "ER71052",
            "cultural": "ER71054", "community": "ER71056",
            "environment": "ER71058", "international": "ER71060",
            "other": "ER71063",
        },
    },
    {
        "year": 2019, "relation_var": "ER34703", "income_var": "ER77448",
        "wealth1_var": "ER77509", "wealth2_var": "ER77511", "itemized_var": "ER73780",
        "volunteer_head_hours_var": "ER72724", "volunteer_spouse_hours_var": "ER72737",
        "hours_kind": "typical_week",
        "giving": {
            "religious": "ER77064", "combo": "ER77066", "needy": "ER77068",
            "health": "ER77070", "education": "ER77072", "youth": "ER77074",
            "cultural": "ER77076", "community": "ER77078",
            "environment": "ER77080", "international": "ER77082",
            "other": "ER77085",
        },
    },
    {
        "year": 2021, "relation_var": "ER34903", "income_var": "ER81775",
        "wealth1_var": "ER81836", "wealth2_var": "ER81838", "itemized_var": "ER79901",
        "volunteer_head_hours_var": "ER78801", "volunteer_spouse_hours_var": "ER78814",
        "hours_kind": "typical_week",
        # "community" was dropped from the questionnaire starting this
        # wave (replaced in part by gating MGROUP1/MGROUP2 questions that
        # are themselves whether-flags, not amounts); no amount variable
        # exists for it from here on.
        "giving": {
            "religious": "ER81340", "combo": "ER81348", "needy": "ER81355",
            "health": "ER81362", "international": "ER81369",
            "education": "ER81377", "youth": "ER81384", "cultural": "ER81391",
            "environment": "ER81398", "other": "ER81416",
        },
    },
    {
        "year": 2023, "relation_var": "ER35103", "income_var": "ER85629",
        "wealth1_var": "ER85690", "wealth2_var": "ER85692", "itemized_var": "ER83870",
        "volunteer_head_hours_var": "ER82794", "volunteer_spouse_hours_var": "ER82807",
        "hours_kind": "typical_week",
        "giving": {
            "religious": "ER85209", "combo": "ER85216", "needy": "ER85223",
            "health": "ER85230", "international": "ER85237",
            "education": "ER85244", "youth": "ER85251", "cultural": "ER85258",
            "environment": "ER85265", "other": "ER85273",
        },
    },
]

_HEAD_RELATION_CODE = 10
_ID_VARS = ("ER30001", "ER30002")
_INFIX_PATTERN = re.compile(r"(?:long |double |str\d* )?(\S+)\s+(\d+)\s*-\s*(\d+)")


def _parse_infix(do_path: str) -> Dict[str, "tuple[int, int]"]:
    with open(do_path, encoding="latin-1") as fh:
        text = fh.read()
    start = text.index("infix")
    end = text.index("using", start)
    block = text[start:end]
    spans = {}
    for match in _INFIX_PATTERN.finditer(block):
        name, col_start, col_end = match.group(1), int(match.group(2)), int(match.group(3))
        spans[name] = (col_start, col_end)
    return spans


def _recode_missing(series: pd.Series, width: int) -> pd.Series:
    """Recode PSID's all-9s (NA/refused) and all-9s-minus-one (DK) sentinels to NaN."""
    na_code = int("9" * width)
    dk_code = int("9" * (width - 1) + "8")
    return series.where(~series.isin([na_code, dk_code]))


def load_psid_philanthropy(data_path: str, do_path: str) -> pd.DataFrame:
    """Load a user-downloaded PSID individual-level cross-year extract into
    a long household-giving/volunteering table.

    The Panel Study of Income Dynamics (PSID) is a public-use survey; its
    Conditions of Use (agreed to when registering at
    https://simba.isr.umich.edu) require: no transfer of downloaded data to
    a third party, no attempt to identify any individual or family, deletion
    of the data after the analysis it was obtained for, reporting
    publications that use it, and acknowledging PSID as the source. This
    reader never downloads, caches, or redistributes PSID data: point it at
    a fixed-width extract and its accompanying Stata ``.do`` file (column
    positions only; no Stata or any other new dependency is used to read
    it) that you obtained yourself from the PSID Data Center under your own
    registration. Never commit PSID rows, or anything derived at the
    row/person level, to this repository or its docs; aggregate statistics
    only.

    This reader is built against a fixed list of PSID ER/S variables (read
    from PSID Data Center job 365729's own ``.do`` file and cross-checked
    against its codebook, covering 2001 to 2023 "Current Year Heads
    Individual Data": one row per person who was Head/Reference Person in
    at least one selected wave). ER/S variable numbers are fixed, study-wide
    PSID identifiers, not job-specific, so a differently-selected extract
    that still includes these variables reads identically. An extract with
    a *smaller* variable selection than job 365729 simply omits some
    columns: a wave whose Head/Reference-person variable is entirely
    absent is skipped outright, while a wave whose Head/Reference-person
    variable is present but another of its variables is missing raises a
    plain ``KeyError``, since that combination means this table and the
    extract disagree about what was selected.

    For each wave, a person counts as that household's Head only when their
    relation-to-head/reference-person code is 10; sequence number is not
    used; the household key is the person's own 1968 interview number and
    person number (``ER30001 * 1000 + ER30002``), which identifies them
    (and so their household, while they are its Head) across every wave.

    Giving amounts, itemized contributions, and volunteering hours all use
    PSID's standard missing-data convention for a field of width *w*: a
    value of *w* nines means Not Applicable/Refused and *w* nines with a
    trailing 8 means Don't Know; both are recoded to ``NaN`` here rather
    than left as large sentinel integers. ``family_income``, ``wealth1``,
    and ``wealth2`` are PSID-generated/imputed totals, not raw survey
    responses, and were directly checked against the real data for this
    convention: they carry no such sentinel values, so this reader leaves
    them unrecoded (a large value there, e.g. a 7-digit income, is real).
    ``total_giving`` sums whatever per-category amounts a wave actually
    asked (``NaN`` only when every category for that wave is ``NaN``, so a
    few Don't-Know categories do not blank the whole total); the categories
    themselves differ by wave, most notably 2001 (five categories plus one
    combined "checkpoint" amount for the rest) and 2021 onward (no
    "community" category).

    Volunteering hours are **three different measures that must never be
    combined into one series**: 2001 asks hours volunteered *last year*,
    split by head and spouse; 2003 and 2005 ask hours volunteered
    "regularly," one battery of up to 14 slots (one per organization type)
    that is not split by head/spouse at all, so this reader returns it as
    a single household-level total; 2017, 2019, 2021 and 2023 ask hours
    volunteered in a *typical week*, again split by head and spouse. 2007
    through 2015 have no hours amount in this extract (only
    whether-volunteered flags, which this reader does not return). The
    three measures are returned as separate columns, ``NaN`` in every wave
    they do not apply to, so a caller cannot accidentally splice them into
    a single trend.

    Parameters
    ----------
    data_path : str
        Path to the fixed-width ASCII data file (PSID calls this the
        ``.txt`` file in a Data Center download).
    do_path : str
        Path to the accompanying Stata ``.do`` file from the same
        download; only its ``infix`` column-position block is read.

    Returns
    -------
    pandas.DataFrame
        Long format, one row per (household, wave actually observed for
        that household's Head). Columns: ``household_key``, ``year``,
        ``total_giving``, one ``giving_<category>`` column per category
        name used anywhere in ``_WAVES`` (``NaN`` in waves that did not ask
        that category), ``itemized_charitable_contrib_amount`` (PSID
        G102A, tax itemizers only), ``family_income``, ``wealth1``
        (without home equity), ``wealth2`` (with home equity),
        ``head_volunteer_hours_annual``, ``spouse_volunteer_hours_annual``,
        ``household_volunteer_hours_regular`` (2003 and 2005 only),
        ``head_volunteer_hours_typical_week``,
        ``spouse_volunteer_hours_typical_week``.

    Notes
    -----
    Source: Panel Study of Income Dynamics, public use dataset, produced
    and distributed by the Survey Research Center, Institute for Social
    Research, University of Michigan, Ann Arbor, MI.
    """
    spans = _parse_infix(do_path)

    category_names: List[str] = sorted(
        {cat for wave in _WAVES for cat in wave["giving"]}
    )

    needed_vars = set(_ID_VARS)
    for wave in _WAVES:
        needed_vars.add(wave["relation_var"])
        for key in ("income_var", "wealth1_var", "wealth2_var", "itemized_var",
                    "volunteer_head_hours_var", "volunteer_spouse_hours_var"):
            if wave[key] is not None:
                needed_vars.add(wave[key])
        needed_vars.update(wave["giving"].values())
        needed_vars.update(wave.get("regular_org_hours_vars", []))

    ordered_vars = [v for v in needed_vars if v in spans]
    colspecs = [(spans[v][0] - 1, spans[v][1]) for v in ordered_vars]
    raw = pd.read_fwf(data_path, colspecs=colspecs, names=ordered_vars)

    household_key = raw[_ID_VARS[0]].astype("int64") * 1000 + raw[_ID_VARS[1]].astype("int64")

    frames = []
    for wave in _WAVES:
        if wave["relation_var"] not in raw.columns:
            # This wave wasn't included in the caller's Data Center job
            # (a smaller variable selection than the one this reader was
            # developed against); nothing to read for it.
            continue
        is_head = raw[wave["relation_var"]] == _HEAD_RELATION_CODE
        if not is_head.any():
            continue
        frame = pd.DataFrame({
            "household_key": household_key[is_head].to_numpy(),
            "year": wave["year"],
        })

        def _amount(var: Optional[str], recode: bool = True) -> Any:
            if var is None:
                return float("nan")
            col = raw.loc[is_head, var].astype("float64")
            if not recode:
                return col.to_numpy()
            width = spans[var][1] - spans[var][0] + 1
            return _recode_missing(col, width).to_numpy()

        for cat in category_names:
            frame[f"giving_{cat}"] = _amount(wave["giving"].get(cat))
        frame["total_giving"] = frame[[f"giving_{c}" for c in category_names]].sum(
            axis=1, skipna=True, min_count=1
        )
        frame["itemized_charitable_contrib_amount"] = _amount(wave["itemized_var"])
        # family_income/wealth1/wealth2 are PSID-generated/imputed totals,
        # not raw survey responses; direct inspection of the real data
        # found zero occurrences of the 9s-sentinel convention on these
        # fields, so (unlike giving/itemized/hours) they are left unrecoded.
        frame["family_income"] = _amount(wave["income_var"], recode=False)
        frame["wealth1"] = _amount(wave["wealth1_var"], recode=False)
        frame["wealth2"] = _amount(wave["wealth2_var"], recode=False)

        for who, var_key in (("head", "volunteer_head_hours_var"), ("spouse", "volunteer_spouse_hours_var")):
            annual_col = f"{who}_volunteer_hours_annual"
            typical_col = f"{who}_volunteer_hours_typical_week"
            if wave["hours_kind"] == "annual":
                frame[annual_col] = _amount(wave[var_key])
                frame[typical_col] = float("nan")
            elif wave["hours_kind"] == "typical_week":
                frame[annual_col] = float("nan")
                frame[typical_col] = _amount(wave[var_key])
            else:
                frame[annual_col] = float("nan")
                frame[typical_col] = float("nan")

        regular_vars = wave.get("regular_org_hours_vars")
        if regular_vars:
            slots = pd.DataFrame({v: _amount(v) for v in regular_vars})
            frame["household_volunteer_hours_regular"] = slots.sum(
                axis=1, skipna=True, min_count=1
            ).to_numpy()
        else:
            frame["household_volunteer_hours_regular"] = float("nan")

        frames.append(frame)

    result = pd.concat(frames, ignore_index=True)
    return result.sort_values(["household_key", "year"]).reset_index(drop=True)
