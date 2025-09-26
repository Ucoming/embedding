"""High level orchestration of the embedding workflow."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .chunking import heuristic_token_counter
from .models.base import EmbeddingBackend
from .storage import IncrementalSaver, iter_part_paths

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    chunk_overlap: int = 50
    checkpoint_chunk_size: int = 10000
    parts_per_directory: int = 200


class EmbeddingProcessor:
    """Coordinate multiple embedding backends to process tabular data."""

    def __init__(
        self,
        *,
        bge_backend: Optional[EmbeddingBackend] = None,
        gpt_backend: Optional[EmbeddingBackend] = None,
        config: ProcessorConfig | None = None,
    ) -> None:
        self.config = config or ProcessorConfig()
        self.backends: Dict[str, EmbeddingBackend] = {}
        if bge_backend is not None:
            self.backends["bge"] = bge_backend
        if gpt_backend is not None:
            self.backends["gpt"] = gpt_backend

    # ------------------------------------------------------------------
    def count_tokens(self, text: str, model_type: str = "bge") -> int:
        backend = self.backends.get(model_type)
        if backend is None:
            if model_type == "gpt":
                return heuristic_token_counter(text)
            raise ValueError(f"Backend '{model_type}' is not initialised")
        return backend.count_tokens(text)

    # ------------------------------------------------------------------
    def process_dataframe(
        self,
        df: pd.DataFrame,
        *,
        text_column: str,
        use_bge: bool = True,
        use_gpt: bool = False,
        batch_size: int = 256,
        skip_length_check: bool = True,
        output_path: Optional[str] = None,
        checkpoint_interval: int = 10000,
    ) -> pd.DataFrame:
        if text_column not in df.columns:
            raise KeyError(f"Column '{text_column}' not found in DataFrame")

        texts = df[text_column].fillna("").astype(str).tolist()
        result_df = df.copy()

        if not skip_length_check:
            self._log_length_statistics(texts, use_bge=use_bge, use_gpt=use_gpt)

        if use_bge and "bge" in self.backends:
            embeddings = self._process_with_checkpoint(
                texts,
                backend_key="bge",
                batch_size=batch_size,
                output_path=output_path,
                checkpoint_interval=checkpoint_interval,
            )
            result_df["bge_embedding"] = [self._serialise_result(item) for item in embeddings]

        if use_gpt and "gpt" in self.backends:
            embeddings = self._process_with_checkpoint(
                texts,
                backend_key="gpt",
                batch_size=batch_size,
                output_path=output_path,
                checkpoint_interval=checkpoint_interval,
            )
            result_df["gpt_embedding"] = [self._serialise_result(item) for item in embeddings]

        return result_df

    # ------------------------------------------------------------------
    def _serialise_result(self, value: Optional[np.ndarray | List[np.ndarray]]):
        if value is None:
            return None
        if isinstance(value, list):
            return [item.tolist() if isinstance(item, np.ndarray) else item for item in value]
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _log_length_statistics(self, texts: List[str], *, use_bge: bool, use_gpt: bool) -> None:
        if use_bge and "bge" in self.backends:
            long_texts = sum(
                1 for text in texts if self.backends["bge"].count_tokens(text) > self.backends["bge"].max_tokens
            )
            logger.info(
                "BGE: %s/%s texts exceed %s tokens",
                long_texts,
                len(texts),
                self.backends["bge"].max_tokens,
            )
        if use_gpt and "gpt" in self.backends:
            long_texts = sum(
                1 for text in texts if self.backends["gpt"].count_tokens(text) > self.backends["gpt"].max_tokens
            )
            logger.info(
                "GPT: %s/%s texts exceed %s tokens",
                long_texts,
                len(texts),
                self.backends["gpt"].max_tokens,
            )

    # ------------------------------------------------------------------
    def _process_with_checkpoint(
        self,
        texts: List[str],
        *,
        backend_key: str,
        batch_size: int,
        output_path: Optional[str],
        checkpoint_interval: int,
    ) -> List[Optional[np.ndarray | List[np.ndarray]]]:
        backend = self.backends[backend_key]
        total_count = len(texts)
        backend_output = self._backend_output_path(output_path, backend_key) if output_path else None
        saver = (
            IncrementalSaver(
                backend_output,
                chunk_size=self.config.checkpoint_chunk_size,
                parts_per_directory=self.config.parts_per_directory,
            )
            if backend_output
            else None
        )

        start_index = 0
        if saver:
            state = saver.load_checkpoint()
            start_index = min(state.processed_count, total_count)
            if start_index:
                logger.info("Skipping the first %s items; already processed.", start_index)

        results: List[Optional[np.ndarray | List[np.ndarray]]] = [None] * total_count
        if start_index:
            existing = self._load_existing_results(backend_output, start_index)
            for idx, value in enumerate(existing):
                if idx < total_count:
                    results[idx] = value

        current_index = start_index
        processed = start_index
        current_batch: List[int] = []

        try:
            while current_index < total_count:
                text = texts[current_index] or ""
                token_length = backend.count_tokens(text)

                if token_length > backend.max_tokens:
                    if current_batch:
                        self._flush_batch(
                            backend,
                            texts,
                            current_batch,
                            results,
                            start_index,
                            batch_size,
                        )
                        processed += len(current_batch)
                        self._maybe_checkpoint(saver, processed, total_count, results, start_index, checkpoint_interval)
                        current_batch = []

                    chunks = backend.split_text(text, chunk_overlap=self.config.chunk_overlap)
                    chunk_embeddings = [backend.embed_batch([chunk], batch_size=1)[0] for chunk in chunks]
                    results[current_index] = chunk_embeddings
                    processed += 1
                    current_index += 1
                    self._maybe_checkpoint(saver, processed, total_count, results, start_index, checkpoint_interval)
                    continue

                current_batch.append(current_index)
                current_index += 1

                if len(current_batch) >= batch_size:
                    self._flush_batch(
                        backend,
                        texts,
                        current_batch,
                        results,
                        start_index,
                        batch_size,
                    )
                    processed += len(current_batch)
                    self._maybe_checkpoint(saver, processed, total_count, results, start_index, checkpoint_interval)
                    current_batch = []

            if current_batch:
                self._flush_batch(
                    backend,
                    texts,
                    current_batch,
                    results,
                    start_index,
                    batch_size,
                )
                processed += len(current_batch)
                current_batch = []
                self._maybe_checkpoint(saver, processed, total_count, results, start_index, checkpoint_interval)

        except KeyboardInterrupt:  # pragma: no cover - interactive behaviour
            logger.warning("Processing interrupted by user; partial progress kept on disk.")
            raise
        finally:
            if saver and processed >= total_count:
                saver.cleanup()
            backend.cleanup()

        return results

    def _flush_batch(
        self,
        backend: EmbeddingBackend,
        texts: List[str],
        batch_indices: List[int],
        results: List[Optional[np.ndarray | List[np.ndarray]]],
        start_index: int,
        batch_size: int,
    ) -> None:
        batch_texts = [texts[i] for i in batch_indices]
        try:
            embeddings = backend.embed_batch(batch_texts, batch_size=min(len(batch_texts), batch_size))
        except RuntimeError as exc:
            if "CUDA out of memory" in str(exc) and len(batch_texts) > 1:
                logger.error("CUDA OOM detected, retrying batch sequentially")
                for index in batch_indices:
                    single_embedding = backend.embed_batch([texts[index]], batch_size=1)[0]
                    results[index - start_index] = single_embedding
                return
            raise

        for local_index, embedding in zip(batch_indices, embeddings):
            results[local_index] = embedding

    def _maybe_checkpoint(
        self,
        saver: Optional[IncrementalSaver],
        processed: int,
        total: int,
        results: List[Optional[np.ndarray | List[np.ndarray]]],
        start_index: int,
        interval: int,
    ) -> None:
        if not saver or processed == 0:
            return
        if processed % interval == 0 or processed == total:
            saver.save_checkpoint(
                processed_count=processed,
                total_count=total,
                results=results,
                start_index=start_index,
            )

    def _backend_output_path(self, output_path: Optional[str], backend_key: str) -> Optional[str]:
        if not output_path:
            return None
        base, ext = os.path.splitext(output_path)
        suffix = f"_{backend_key}"
        if ext:
            return f"{base}{suffix}{ext}"
        return f"{output_path}{suffix}"

    def _load_existing_results(
        self, backend_output: Optional[str], processed_count: int
    ) -> List[Optional[np.ndarray | List[np.ndarray]]]:
        if not backend_output or processed_count <= 0:
            return []
        embeddings_dir = os.path.join(os.path.dirname(backend_output), "embeddings")
        part_files = list(iter_part_paths(embeddings_dir))
        if not part_files:
            return []

        collected: List[Optional[np.ndarray | List[np.ndarray]]] = []
        for file_path in part_files:
            try:
                if file_path.endswith(".parquet"):
                    df = pd.read_parquet(file_path)
                    collected.extend(df.get("embedding_result", []).tolist())
                else:
                    with open(file_path, "r", encoding="utf-8") as handle:
                        data = json.load(handle)
                    collected.extend(data if isinstance(data, list) else [data])
            except Exception as exc:  # pragma: no cover - IO issues
                logger.warning("Failed to load %s: %s", file_path, exc)
                continue
            if len(collected) >= processed_count:
                break

        return collected[:processed_count]
