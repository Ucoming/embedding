"""Implementation of the BGE/SentenceTransformer embedding backend."""
from __future__ import annotations

import logging
import os
from typing import List, Sequence

import numpy as np

try:
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer
    import torch
except ImportError:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    torch = None  # type: ignore

from .base import EmbeddingBackend

logger = logging.getLogger(__name__)


class BGEEmbeddingBackend(EmbeddingBackend):
    """SentenceTransformer based embedding backend."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-Embedding-8B",
        *,
        max_tokens: int = 32000,
        local_model_dir: str | None = None,
        default_model_location: str | None = None,
        chunk_overlap: int = 50,
    ) -> None:
        if SentenceTransformer is None or AutoTokenizer is None:
            raise RuntimeError(
                "sentence_transformers and transformers must be installed to use the BGE backend"
            )

        self.name = model_name
        self.max_tokens = max_tokens
        self.chunk_overlap = chunk_overlap

        local_dir = local_model_dir or os.path.join(os.getcwd(), "model")
        os.makedirs(local_dir, exist_ok=True)
        self._model_dir = local_dir

        device = "cuda" if torch and torch.cuda.is_available() else "cpu"
        logger.info("Loading BGE model %s on %s", model_name, device)

        os.environ.setdefault("TRANSFORMERS_CACHE", local_dir)
        os.environ.setdefault("HF_HOME", local_dir)

        local_model_path = _resolve_model_location(
            model_name=model_name,
            cache_dir=local_dir,
            default_location=default_model_location,
        )
        load_path = local_model_path if os.path.exists(local_model_path) else model_name

        model_kwargs = {
            "trust_remote_code": True,
            "device": device,
            "model_kwargs": {
                "attn_implementation": "eager",
            },
        }
        if torch:
            model_kwargs["model_kwargs"]["torch_dtype"] = torch.bfloat16
        if load_path == model_name:
            model_kwargs["cache_folder"] = local_dir

        self._model = SentenceTransformer(load_path, **model_kwargs)
        self._tokenizer = AutoTokenizer.from_pretrained(
            load_path,
            trust_remote_code=True,
            cache_dir=local_dir,
        )

        if device == "cuda" and torch:
            logger.info("BGE model loaded on GPU: %s", torch.cuda.get_device_name())
        else:
            logger.info("BGE model loaded on CPU")

    def count_tokens(self, text: str) -> int:
        tokens = self._tokenizer.encode(text, add_special_tokens=True)
        return len(tokens)

    def embed_batch(self, texts: Sequence[str], *, batch_size: int = 1) -> List[np.ndarray]:
        show_progress = batch_size > 1
        embeddings = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
            batch_size=batch_size,
        )
        if isinstance(embeddings, np.ndarray):
            return [embeddings[i] for i in range(len(texts))]
        return list(embeddings)

    def supports_batching(self) -> bool:
        return True

    def cleanup(self) -> None:
        if torch and torch.cuda.is_available():  # pragma: no cover - hardware specific
            torch.cuda.empty_cache()


def _resolve_model_location(*, model_name: str, cache_dir: str, default_location: str | None) -> str:
    if default_location:
        if os.path.isabs(default_location):
            return default_location
        return os.path.join(cache_dir, default_location)

    safe_name = model_name.split(":", 1)[0].split("/")[-1]
    return os.path.join(cache_dir, safe_name)
