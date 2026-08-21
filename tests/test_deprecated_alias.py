"""The SolicitationWindowTransformer alias warns but still resolves."""

import pytest

from philanthropy import preprocessing
from philanthropy.preprocessing import DischargeToSolicitationWindowTransformer


def test_alias_warns_and_is_the_canonical_class():
    with pytest.warns(DeprecationWarning, match="deprecated alias"):
        alias = preprocessing.SolicitationWindowTransformer
    assert alias is DischargeToSolicitationWindowTransformer


def test_unknown_attribute_still_raises_attribute_error():
    with pytest.raises(AttributeError):
        preprocessing.NoSuchTransformer
