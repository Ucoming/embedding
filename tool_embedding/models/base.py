"""Base classes and protocols for embedding backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, List, Sequence
import numpy as np


@dataclass
class EmbeddingResult:
    """Represents an embedding for a single logical document."""

    vectors: List[np.ndarray]

    @classmethod
    def from_array(cls, array: np.ndarray) -> "EmbeddingResult":
        return cls(vectors=[array])

    def to_serialisable(self) -> List[List[float]]:
        return [vector.tolist() for vector in self.vectors]


class EmbeddingBackend(ABC):
    """Abstract base class for embedding providers."""

    name: str
    max_tokens: int

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """Return the number of tokens for *text*."""

    @abstractmethod
    def embed_batch(self, texts: Sequence[str], *, batch_size: int = 1) -> List[np.ndarray]:
        """Return embeddings for *texts*.

        The result must contain one vector per text in the same order.
        """

    def supports_batching(self) -> bool:
        return True

    def cleanup(self) -> None:
        """Release provider specific resources if necessary."""

    def split_text(self, text: str, chunk_overlap: int = 0) -> List[str]:
        """Split the text using the backend specific token counting."""
        from ..chunking import split_text_by_tokens

        return split_text_by_tokens(
            text,
            self.max_tokens,
            counter=self.count_tokens,
            chunk_overlap=chunk_overlap,
        )
