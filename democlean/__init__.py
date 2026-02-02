"""democlean: Score robot demonstrations by motion quality."""

__version__ = "0.2.0"

from democlean.embeddings import Encoder, RawEmbedding, get_encoder
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
    "Encoder",
    "RawEmbedding",
    "get_encoder",
    "estimate_mi",
    "estimate_mi_ksg",
    "estimate_mi_with_ci",
    "reduce_dimensions",
]
