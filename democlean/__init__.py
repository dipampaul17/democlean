"""democlean: Score robot demonstrations by motion quality."""

__version__ = "0.1.4"

from democlean.mi import (
    estimate_mi,
    estimate_mi_ksg,
    estimate_mi_with_ci,
    reduce_dimensions,
)
from democlean.scorer import DemoScorer, EpisodeScore

__all__ = [
    "DemoScorer",
    "EpisodeScore",
    "estimate_mi",
    "estimate_mi_ksg",
    "estimate_mi_with_ci",
    "reduce_dimensions",
]
