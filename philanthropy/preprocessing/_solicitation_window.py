"""
philanthropy.preprocessing._solicitation_window
================================================
Deprecated alias for :class:`DischargeToSolicitationWindowTransformer`.

Two public names for one transformer inflate the API surface without adding
capability, so the alias goes away in 1.0.0. It resolves to the same class
object rather than a subclass, so ``isinstance`` and ``clone`` behave exactly as
before; the deprecation warning is emitted on attribute access by the
subpackage's module-level ``__getattr__`` (PEP 562).
"""

from ._discharge_window import DischargeToSolicitationWindowTransformer

#: Deprecated. Use ``DischargeToSolicitationWindowTransformer``.
SolicitationWindowTransformer = DischargeToSolicitationWindowTransformer
