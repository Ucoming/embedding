"""Utilities for incremental storage and checkpoint handling."""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime

from typing import Any, Dict, Iterable, List, Sequence, Tuple


import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CheckpointState:
    processed_count: int
    metadata: Dict[str, Any]


def _serialise_embedding(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_serialise_embedding(item) for item in value]
    return value


class IncrementalSaver:
    """Handles saving embeddings in parts with resumable checkpoints."""


    def __init__(
        self,
        output_path: str,
        *,
        chunk_size: int = 10000,
        parts_per_directory: int = 200,
    ) -> None:
        self.output_path = output_path
        self.chunk_size = chunk_size
        self.parts_per_directory = max(parts_per_directory, 1)



    # ------------------------------------------------------------------
    # Checkpoint helpers
    def _checkpoint_file(self) -> str:
        base_path = os.path.splitext(self.output_path)[0]
        return f"{base_path}_checkpoint.json"

    def _embeddings_dir(self) -> str:
        return os.path.join(os.path.dirname(self.output_path), "embeddings")

    def load_checkpoint(self) -> CheckpointState:
        checkpoint_path = self._checkpoint_file()
        if not os.path.exists(checkpoint_path):
            return CheckpointState(processed_count=0, metadata={})

        with open(checkpoint_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)

        processed_count = data.get("processed_count", 0)
        embeddings_dir = self._embeddings_dir()

        part_files = _discover_part_files(embeddings_dir)
        if part_files:
            max_part = max(index for index, _ in part_files)
            estimated = (max_part + 1) * self.chunk_size
            if abs(estimated - processed_count) > self.chunk_size:
                logger.warning(
                    "Checkpoint count (%s) differs from part files (%s). Using part file count.",
                    processed_count,
                    estimated,
                )
                processed_count = estimated


        logger.info(
            "Resuming from checkpoint: %s items processed (%.1f%%)",
            processed_count,
            data.get("progress_percent", 0.0),
        )
        return CheckpointState(processed_count=processed_count, metadata=data)

    def save_checkpoint(
        self,
        *,
        processed_count: int,
        total_count: int,
        results: Sequence[Any],
        start_index: int,
    ) -> None:
        checkpoint_path = self._checkpoint_file()
        checkpoint_data = {
            "timestamp": datetime.now().isoformat(),
            "processed_count": processed_count,
            "total_count": total_count,
            "progress_percent": (processed_count / total_count * 100) if total_count else 0,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as handle:
            json.dump(checkpoint_data, handle, indent=2, ensure_ascii=False)

        if not results or processed_count <= 0:
            return

        embeddings_dir = self._embeddings_dir()
        os.makedirs(embeddings_dir, exist_ok=True)

        current_chunk = (processed_count - 1) // self.chunk_size
        abs_start = current_chunk * self.chunk_size
        abs_end = min(processed_count, (current_chunk + 1) * self.chunk_size)

        if len(results) >= abs_end:
            rel_start = abs_start
            rel_end = abs_end
        else:
            rel_start = abs_start - start_index
            rel_end = abs_end - start_index

        if rel_start < 0 or rel_start >= len(results):
            return

        if processed_count % self.chunk_size != 0 and processed_count != total_count:
            # We only persist when the chunk is complete or when processing
            # finished to avoid partial files.
            return

        chunk_results = results[rel_start:rel_end]
        if not chunk_results:
            return

        serialised = [_serialise_embedding(item) for item in chunk_results]

        chunk_dir = self._chunk_directory(current_chunk)
        os.makedirs(chunk_dir, exist_ok=True)
        chunk_base = os.path.join(chunk_dir, f"part-{current_chunk:05d}")

        file_path = self._write_chunk(chunk_base, serialised)
        logger.info("Saved %s embeddings to %s", len(chunk_results), file_path)

    def cleanup(self) -> None:
        checkpoint_path = self._checkpoint_file()
        temp_output = f"{os.path.splitext(self.output_path)[0]}_temp.parquet"
        for path in (checkpoint_path, temp_output):
            if os.path.exists(path):
                os.remove(path)

    def _write_chunk(self, base_path: str, data: Sequence[Any]) -> str:
        parquet_path = f"{base_path}.parquet"
        tmp_path = f"{parquet_path}.tmp"
        df = pd.DataFrame({"embedding_result": list(data)})
        try:
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, parquet_path)
            return parquet_path
        except ImportError:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            json_path = f"{base_path}.json"
            with open(json_path, "w", encoding="utf-8") as handle:
                json.dump(list(data), handle, ensure_ascii=False)
            return json_path


    # ------------------------------------------------------------------
    def _chunk_directory(self, chunk_index: int) -> str:
        if self.parts_per_directory <= 0:
            return self._embeddings_dir()
        group = chunk_index // self.parts_per_directory
        return os.path.join(self._embeddings_dir(), f"group-{group:05d}")


def _discover_part_files(directory: str) -> List[Tuple[int, str]]:
    """Return sorted ``(index, path)`` tuples for embedding part files."""

    if not directory or not os.path.exists(directory):
        return []

    part_files: List[Tuple[int, str]] = []
    for root, _, files in os.walk(directory):
        for filename in files:
            if not filename.startswith("part-"):
                continue
            ext = os.path.splitext(filename)[1]
            if ext not in {".parquet", ".json"}:
                continue
            try:
                chunk_index = int(filename.split("-")[1].split(".")[0])
            except (IndexError, ValueError):
                continue
            part_files.append((chunk_index, os.path.join(root, filename)))

    part_files.sort(key=lambda item: item[0])
    return part_files


def iter_part_paths(directory: str) -> Iterable[str]:
    """Yield sorted embedding part file paths under ``directory``."""

    for _, path in _discover_part_files(directory):
        yield path

