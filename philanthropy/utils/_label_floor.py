"""
philanthropy.utils._label_floor
===============================
Minimum label counts before a model is worth fitting on a file.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

# Counts of the rarer class in the training labels. Each was read off a
# learning curve on a validation fold (no published test fold used),
# training subsampled to N rarer-class labels at the file's own base rate,
# 10 seeds, scored on both ROC-AUC and the top-10% hit rate the Results
# pages use. The best rule is the one with the highest top-10% hit rate.
#
# lapse: PSID households with 2+ giving waves, train waves < 2013, validate
#   wave 2013 (20% lapse). From 100 lapsed households up every draw beat the
#   best rule on both measures (top 10%: 48 vs 42 of 100 at 100 labels).
#   On DonorsChoose multi-year donors (validate FY2014), ROC-AUC reached
#   the rule in every draw only at 200 retained donors, and the top-10% hit
#   rate in every draw only on all training rows (80.5 vs 79.2): there the
#   model ties the rule at the top of the list at any label count, which is
#   what the Lapse page reports. A floor is necessary, not sufficient.
# major_gift: KDD Cup 1998 response (5% positive), 55/15/30 split. At 800
#   positives DonorPropensityModel beat the rule in 10 of 10 draws on both
#   measures and MajorGiftClassifier in 10 of 10 on ROC-AUC and 9 of 10 on
#   the top 10%; at 400, MajorGiftClassifier lost on both in every draw.
#   On PSID's real $1,000 upgrade label (train waves < 2013, validate wave
#   2013, 16% positive) MajorGiftClassifier, the leadership model's
#   backend, beat the best rule's top-10% in every draw from 800 positives
#   (39 vs 35 of 100) but not at 400 (6 of 10); its ROC-AUC matched the rule
#   in 7 of 10 draws at 800 and in every draw only on all 2,346.
#   DonorPropensityModel cleared both measures from 200.
LABEL_FLOORS: Dict[str, int] = {"lapse": 100, "major_gift": 800}


def check_label_floor(y: Any, task: str) -> str:
    """Say whether a file has enough labels to fit a model for ``task``.

    Counts the rarer of the two classes in ``y`` (lapsed vs retained for
    lapse, major donors vs everyone else for major gift) and compares it to
    ``LABEL_FLOORS[task]``. Below the floor, at least one training draw in
    ten scored below the simple rule on held-out data (lapse), or most did
    (major gift), so the rule is the safer choice.

    Parameters
    ----------
    y : array-like of shape (n_samples,)
        Binary training labels.
    task : {"lapse", "major_gift"}
        Which floor to apply.

    Returns
    -------
    str
        ``"run"`` or ``"not enough labels"``.

    Raises
    ------
    ValueError
        If ``task`` is unknown, ``y`` has missing labels, or ``y`` has more
        than two classes.

    Examples
    --------
    >>> from philanthropy.utils import check_label_floor
    >>> check_label_floor([1] * 150 + [0] * 850, "lapse")
    'run'
    >>> check_label_floor([1] * 150 + [0] * 850, "major_gift")
    'not enough labels'
    """
    if task not in LABEL_FLOORS:
        raise ValueError(
            f"`task` must be one of {sorted(LABEL_FLOORS)}, got {task!r}."
        )
    y = np.asarray(y).ravel()
    if pd.isna(y).any():
        raise ValueError(
            "`y` has missing labels; drop or fill the unlabelled rows first."
        )
    _, counts = np.unique(y, return_counts=True)
    if len(counts) > 2:
        raise ValueError(f"`y` must be binary, got {len(counts)} classes.")
    rarer = counts.min() if len(counts) == 2 else 0
    return "run" if rarer >= LABEL_FLOORS[task] else "not enough labels"
