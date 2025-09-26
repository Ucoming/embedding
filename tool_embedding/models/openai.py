"""OpenAI compatible embedding backend."""
from __future__ import annotations

import logging
import os
import time
from typing import List, Sequence, Tuple

import numpy as np

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

from .base import EmbeddingBackend
from ..chunking import heuristic_token_counter

logger = logging.getLogger(__name__)


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """Embedding backend that uses the OpenAI compatible API."""

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        *,
        max_tokens: int = 8192,
        chunk_overlap: int = 50,
        env_file: str | None = "oai_embeddings.env",
    ) -> None:
        if OpenAI is None:
            raise RuntimeError("openai package must be installed to use the OpenAI backend")

        self.name = model_name
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap

        api_key, base_url = self._load_credentials(env_file)
        self._client = OpenAI(api_key=api_key, base_url=base_url)
        logger.info("OpenAI client initialised for model %s", model_name)

    @staticmethod
    def _load_credentials(env_file: str | None) -> Tuple[str, str | None]:
        if env_file and os.path.exists(env_file):
            from dotenv import load_dotenv

            load_dotenv(env_file, override=True)

        api_key = os.getenv("api_key") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("base_url") or os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise RuntimeError("OpenAI API key not found in environment variables")
        return api_key, base_url

    def count_tokens(self, text: str) -> int:
        return heuristic_token_counter(text)

    def supports_batching(self) -> bool:
        return True

    def embed_batch(self, texts: Sequence[str], *, batch_size: int = 1) -> List[np.ndarray]:
        # The OpenAI API accepts a batch of strings. We respect the caller
        # provided batch_size by chunking the request list accordingly.
        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            response = self._client.embeddings.create(input=batch_texts, model=self.name)
            for item in response.data:
                embeddings.append(np.array(item.embedding))
            time.sleep(0.05)
        return embeddings
