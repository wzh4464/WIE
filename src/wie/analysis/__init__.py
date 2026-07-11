"""Post-hoc analysis of window-level influence trajectories.

Currently exposes the RQ2 temporal-pattern classifier
(:mod:`wie.analysis.temporal_patterns`).
"""

from .temporal_patterns import (  # noqa: F401
    PATTERN_LABELS,
    standardize_per_epoch,
    classify_patterns,
    pattern_distribution,
    load_influence_matrix,
)

__all__ = [
    "PATTERN_LABELS",
    "standardize_per_epoch",
    "classify_patterns",
    "pattern_distribution",
    "load_influence_matrix",
]
