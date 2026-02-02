"""Raw embedding backend (passthrough)."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from democlean.embeddings.base import Encoder


class RawEmbedding(Encoder):
    """Passthrough encoder - returns inputs unchanged.

    This is the default encoder, matching democlean's original behavior
    where MI is computed directly on raw state/action features.
    """

    @property
    def name(self) -> str:
        return "raw"

    def encode_states(self, states: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return states unchanged."""
        return states

    def encode_actions(self, actions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Return actions unchanged."""
        return actions
