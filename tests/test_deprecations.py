"""Every live deprecation shim, and a meta-test so none ships untested.

RELEASING.md prescribes this file: an inline ``warnings.warn(...,
DeprecationWarning)`` for a parameter being retired, plus "a registry meta-test
[that] fails when a shim ships untested". `DEPRECATIONS` below is that registry.
Add a row when you deprecate something and the meta-test will make you write the
case for it.
"""

import warnings

import numpy as np
import pytest

from philanthropy.preprocessing import WealthScreeningImputerKNN

# (id, removed_in, callable that should emit exactly one DeprecationWarning)
DEPRECATIONS = [
    (
        "WealthScreeningImputerKNN.group_col_idx",
        "0.8.0",
        lambda: WealthScreeningImputerKNN(
            strategy="knn", n_neighbors=3, add_indicator=False, group_col_idx=1
        ).fit(_two_group_X()),
    ),
]


def _two_group_X():
    return np.column_stack([
        np.r_[np.nan, np.linspace(4e4, 6e4, 19)],
        np.zeros(20),
    ])


@pytest.mark.parametrize("dep_id,removed_in,trigger", DEPRECATIONS,
                         ids=[d[0] for d in DEPRECATIONS])
def test_shim_emits_deprecation_warning(dep_id, removed_in, trigger):
    with pytest.warns(DeprecationWarning) as record:
        trigger()
    messages = [str(w.message) for w in record]
    assert len(messages) == 1, f"{dep_id} emitted {len(messages)} warnings"
    msg = messages[0]
    # A deprecation the caller cannot act on is just noise, so require the
    # message to say when it goes and what to do instead.
    assert removed_in in msg, f"{dep_id} does not say it is removed in {removed_in}"
    assert "deprecated" in msg.lower()


@pytest.mark.parametrize("dep_id,removed_in,trigger", DEPRECATIONS,
                         ids=[d[0] for d in DEPRECATIONS])
def test_shim_still_works(dep_id, removed_in, trigger):
    # A shim exists to buy a migration window. If it raises, it is not a shim.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        trigger()


def test_no_warning_when_the_deprecated_parameter_is_untouched():
    # The other half of the contract: callers who never used it see nothing.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        WealthScreeningImputerKNN(
            strategy="knn", n_neighbors=3, add_indicator=False
        ).fit(_two_group_X())
    assert not [w for w in caught if issubclass(w.category, DeprecationWarning)]


def test_registry_covers_every_deprecation_warning_in_the_package():
    """The meta-test RELEASING.md asks for.

    Walks the package AST for ``warnings.warn(..., DeprecationWarning)`` calls
    and fails if there are more than the registry knows about, so a shim added
    without a row here cannot ship untested. AST rather than grep, so a docstring
    that merely mentions ``DeprecationWarning`` is not miscounted as a shim.
    """
    import ast
    import pathlib

    import philanthropy

    root = pathlib.Path(philanthropy.__file__).parent
    sites = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            name = getattr(func, "attr", getattr(func, "id", None))
            if name != "warn":
                continue
            args = list(node.args) + [kw.value for kw in node.keywords]
            if any(isinstance(a, ast.Name) and a.id == "DeprecationWarning"
                   for a in args):
                sites.append(f"{path.relative_to(root)}:{node.lineno}")

    assert len(sites) == len(DEPRECATIONS), (
        f"{len(sites)} DeprecationWarning call site(s) in the package but "
        f"{len(DEPRECATIONS)} registry row(s). Sites: {sites}. Add a row to "
        "DEPRECATIONS (or remove the shim) so it cannot ship untested."
    )
