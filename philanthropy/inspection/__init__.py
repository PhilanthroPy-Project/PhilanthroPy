"""
philanthropy.inspection
========================
Model-agnostic interpretability helpers for donor-scoring estimators.
"""

from ._importance import donor_feature_importance

__all__ = ["donor_feature_importance"]
