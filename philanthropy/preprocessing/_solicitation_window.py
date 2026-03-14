"""
philanthropy.preprocessing._solicitation_window
================================================
Backward-compatibility alias for DischargeToSolicitationWindowTransformer.
"""

from ._discharge_window import DischargeToSolicitationWindowTransformer

# Backward compatibility: CI / older __init__.py may import SolicitationWindowTransformer
SolicitationWindowTransformer = DischargeToSolicitationWindowTransformer
