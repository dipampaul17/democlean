"""Base encoder interface for embedding backends."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray


class Encoder(ABC):
    """Abstract base class for state/action encoders.

    Encoders transform raw state and action arrays into embeddings
    before mutual information estimation.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Encoder name for logging."""

    @property
    def output_dim(self) -> int | None:
        """Output embedding dimension, or None if variable."""
        return None

    @abstractmethod
    def encode_states(self, states: NDArray[np.floating]) -> NDArray[np.floating]:
        """Encode state observations.

        Args:
            states: Shape (T, state_dim)

        Returns:
            Encoded states, shape (T, embed_dim)
        """

    @abstractmethod
    def encode_actions(self, actions: NDArray[np.floating]) -> NDArray[np.floating]:
        """Encode actions.

        Args:
            actions: Shape (T, action_dim)

        Returns:
            Encoded actions, shape (T, embed_dim)
        """

    def encode_episode(
        self,
        states: NDArray[np.floating],
        actions: NDArray[np.floating],
    ) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Encode full episode (states and actions).

        Default implementation calls encode_states and encode_actions.
        Subclasses may override for joint encoding.
        """
        return self.encode_states(states), self.encode_actions(actions)
