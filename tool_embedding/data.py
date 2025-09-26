"""Data loading helpers."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def load_meeting_data(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
    """Load tabular meeting data from Excel, CSV or Parquet files."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        if path.suffix.lower() == ".xlsx":
            return pd.read_excel(path, sheet_name=sheet_name)
        if path.suffix.lower() == ".csv":
            return pd.read_csv(path)
        if path.suffix.lower() == ".parquet":
            return pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - delegated to pandas
        logger.error("Failed to load %s: %s", file_path, exc)
        raise

    raise ValueError(f"Unsupported file format: {file_path}")
