"""Embeddings for state/action encoding."""

from democlean.embeddings.base import Encoder
from democlean.embeddings.raw import RawEmbedding

__all__ = ["Encoder", "RawEmbedding", "get_encoder"]


def get_encoder(name: str, **kwargs) -> Encoder:
    """Factory function for encoder backends.

    Args:
        name: Encoder name ("raw" or "hpt")
        **kwargs: Passed to encoder constructor

    Returns:
        Encoder instance

    Raises:
        ValueError: If encoder name is unknown
        ImportError: If encoder dependencies are not installed
    """
    if name == "raw":
        return RawEmbedding()
    elif name == "hpt":
        from democlean.embeddings.hpt import HPTEmbedding
        return HPTEmbedding(**kwargs)
    else:
        available = ["raw", "hpt"]
        raise ValueError(f"Unknown encoder: {name}. Available: {available}")
