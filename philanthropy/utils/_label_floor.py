"""
philanthropy.utils._label_floor
===============================
Minimum label counts before a model is worth fitting on a file.
"""

from typing import Any, Dict

import numpy as np
import pandas as pd

# Counts of the rarer class in the training labels. Each was read off a
# learning curve on a validation fold (test split untouched), training
# subsampled to N rarer-class labels at the file's own base rate, 10 seeds:
#
# lapse: DonorsChoose (ICPSR 37898), train FY < 2017, validate FY2017. Below
#   100 retained donors at least one draw in ten scored below the best simple
#   rule's ROC-AUC; from 100 up, every draw matched or beat it.
# major_gift: KDD Cup 1998 response (5% positive), 55/15/30 split. Both
#   DonorPropensityModel and MajorGiftClassifier stayed below the best rule
#   in every seed up to 400 positives and reached it only at about 800.
#   This is a weak-signal file, so treat 800 as conservative.
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
