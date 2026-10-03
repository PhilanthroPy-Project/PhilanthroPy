"""
tests/test_label_floor.py
"""

import numpy as np
import pytest

from philanthropy.utils import LABEL_FLOORS, check_label_floor


@pytest.mark.parametrize("task", sorted(LABEL_FLOORS))
def test_floor_is_inclusive_on_the_rarer_class(task):
    floor = LABEL_FLOORS[task]
    assert check_label_floor([1] * floor + [0] * (floor * 5), task) == "run"
    assert check_label_floor([1] * (floor - 1) + [0] * (floor * 5), task) == "not enough labels"
    # The rarer class binds, whichever label it is.
    assert check_label_floor([0] * (floor - 1) + [1] * (floor * 5), task) == "not enough labels"


def test_single_class_is_not_enough():
    assert check_label_floor(np.ones(10_000), "lapse") == "not enough labels"


def test_bad_task_and_multiclass_raise():
    with pytest.raises(ValueError, match="task"):
        check_label_floor([0, 1], "upgrade")
    with pytest.raises(ValueError, match="binary"):
        check_label_floor([0, 1, 2], "lapse")
