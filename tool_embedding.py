"""Backward compatible entry point for the refactored embedding toolkit."""
from __future__ import annotations

from importlib import import_module as _import_module

_pkg = _import_module("tool_embedding.__init__")

EmbeddingProcessor = _pkg.EmbeddingProcessor  # type: ignore[attr-defined]
ProcessorConfig = _pkg.ProcessorConfig  # type: ignore[attr-defined]
create_processor = _pkg.create_processor  # type: ignore[attr-defined]
load_meeting_data = _pkg.load_meeting_data  # type: ignore[attr-defined]
quick_embedding_pipeline = _pkg.quick_embedding_pipeline  # type: ignore[attr-defined]

__all__ = [
    "EmbeddingProcessor",
    "ProcessorConfig",
    "create_processor",
    "load_meeting_data",
    "quick_embedding_pipeline",
]
