"""Public API for the embedding tool package."""
from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

from .data import load_meeting_data
from .pipeline import EmbeddingProcessor, ProcessorConfig
from .storage import iter_part_paths

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProcessor",
    "ProcessorConfig",
    "create_processor",
    "load_meeting_data",
    "quick_embedding_pipeline",
]


def create_processor(
    *,
    bge_model_name: str = "Qwen/Qwen3-Embedding-8B",
    bge_model_location: Optional[str] = None,
    gpt_model_name: str = "text-embedding-3-small",
    max_bge_tokens: int = 32000,
    max_gpt_tokens: int = 8192,
    chunk_overlap: int = 50,
    checkpoint_chunk_size: int = 10000,
    parts_per_directory: int = 200,
    load_bge: bool = True,
    load_gpt: bool = False,
    local_model_dir: Optional[str] = None,
    env_file: Optional[str] = "oai_embeddings.env",
) -> EmbeddingProcessor:
    """Create an :class:`EmbeddingProcessor` with optional backends."""

    bge_backend = None
    gpt_backend = None

    if load_bge:
        try:
            from .models.bge import BGEEmbeddingBackend

            bge_backend = BGEEmbeddingBackend(
                model_name=bge_model_name,
                max_tokens=max_bge_tokens,
                local_model_dir=local_model_dir,
                default_model_location=bge_model_location,
                chunk_overlap=chunk_overlap,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("Failed to initialise BGE backend: %s", exc)
            raise

    if load_gpt:
        try:
            from .models.openai import OpenAIEmbeddingBackend

            gpt_backend = OpenAIEmbeddingBackend(
                model_name=gpt_model_name,
                max_tokens=max_gpt_tokens,
                chunk_overlap=chunk_overlap,
                env_file=env_file,
            )
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.error("Failed to initialise OpenAI backend: %s", exc)
            raise

    processor = EmbeddingProcessor(
        bge_backend=bge_backend,
        gpt_backend=gpt_backend,
        config=ProcessorConfig(
            chunk_overlap=chunk_overlap,
            checkpoint_chunk_size=checkpoint_chunk_size,
            parts_per_directory=parts_per_directory,
        ),
    )

    return processor


def quick_embedding_pipeline(
    *,
    data_path: str,
    output_path: str,
    text_column: str,
    sheet_name: Optional[str] = None,
    use_bge: bool = True,
    use_gpt: bool = False,
    bge_model_name: str = "Qwen/Qwen3-Embedding-8B",
    bge_model_location: Optional[str] = None,
    gpt_model_name: str = "text-embedding-3-small",
    max_bge_tokens: int = 32000,
    max_gpt_tokens: int = 8192,
    local_model_dir: Optional[str] = None,
    env_file: Optional[str] = "oai_embeddings.env",
    batch_size: int = 64,
    checkpoint_interval: int = 10000,
    checkpoint_chunk_size: int = 10000,
    parts_per_directory: int = 200,
    skip_length_check: bool = True,
) -> pd.DataFrame:
    """High level helper mirroring the behaviour of the legacy script."""

    df = load_meeting_data(data_path, sheet_name=sheet_name)
    processor = create_processor(
        bge_model_name=bge_model_name,
        bge_model_location=bge_model_location,
        gpt_model_name=gpt_model_name,
        max_bge_tokens=max_bge_tokens,
        max_gpt_tokens=max_gpt_tokens,
        chunk_overlap=50,
        checkpoint_chunk_size=checkpoint_chunk_size,
        parts_per_directory=parts_per_directory,
        load_bge=use_bge,
        load_gpt=use_gpt,
        local_model_dir=local_model_dir,
        env_file=env_file,
    )

    result_df = processor.process_dataframe(
        df,
        text_column=text_column,
        use_bge=use_bge,
        use_gpt=use_gpt,
        batch_size=batch_size,
        skip_length_check=skip_length_check,
        output_path=output_path,
        checkpoint_interval=checkpoint_interval,
    )

    embeddings_dir = os.path.join(os.path.dirname(output_path), "embeddings")
    if os.path.exists(embeddings_dir):
        part_files = list(iter_part_paths(embeddings_dir))
        part_dirs = {os.path.dirname(path) for path in part_files}
        logger.info(
            "Saved %s part files across %s directories under %s",
            len(part_files),
            len(part_dirs) or 1,
            embeddings_dir,
        )
    else:
        logger.info("No embeddings directory created; did you supply an output path?")

    return result_df
