"""
philanthropy.preprocessing
==========================
CRM data cleaning, Fiscal Year-aware feature engineering, and
clinical-encounter feature engineering for medical philanthropy.
"""

from typing import Any

from ._transformers import FiscalYearTransformer, CRMCleaner
from ._wealth import WealthScreeningImputer
from ._encounters import EncounterTransformer
from ._rfm import RFMTransformer
from ._planned_giving import PlannedGivingSignalTransformer
from ._grateful_patient import GratefulPatientFeaturizer
from ._discharge_window import DischargeToSolicitationWindowTransformer
from ._wealth_percentile import WealthPercentileTransformer
from ._encounter_recency import EncounterRecencyTransformer
from ._share_of_wallet import WealthScreeningImputerKNN, ShareOfWalletScorer
from ._matching_gift import MatchingGiftFeaturizer

__all__ = [
    "FiscalYearTransformer",
    "CRMCleaner",
    "WealthScreeningImputer",
    "EncounterTransformer",
    "RFMTransformer",
    "PlannedGivingSignalTransformer",
    "GratefulPatientFeaturizer",
    "DischargeToSolicitationWindowTransformer",
    "SolicitationWindowTransformer",
    "WealthPercentileTransformer",
    "EncounterRecencyTransformer",
    "WealthScreeningImputerKNN",
    "ShareOfWalletScorer",
    "MatchingGiftFeaturizer",
]


# Deprecated aliases resolve through __getattr__ (PEP 562) to the same class
# object as the canonical name — not a subclass — so isinstance/clone behave
# as before. They are removed in 1.0.0. No separate alias modules: those would
# be unimported dead code (the alias never loads them).
_DEPRECATED_ALIASES = {
    # alias -> canonical attribute name in this module's globals()
    "SolicitationWindowTransformer": "DischargeToSolicitationWindowTransformer",
}


def __getattr__(name: str) -> Any:
    """Resolve deprecated aliases lazily, warning once per access (PEP 562).

    Kept as an alias rather than a subclass on purpose: a subclass would be a
    different class object, which changes ``type()`` for existing callers and
    would need its own ``get_feature_names_out`` to satisfy the public-API
    contract test. This way the object handed back is the canonical class.
    """
    canonical = _DEPRECATED_ALIASES.get(name)
    if canonical is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import warnings

    warnings.warn(
        f"{name} is a deprecated alias for {canonical} and is removed in 1.0.0. "
        f"Import {canonical} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return globals()[canonical]
