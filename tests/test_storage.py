import json
from pathlib import Path

import numpy as np
import pandas as pd

from tool_embedding.storage import IncrementalSaver


def test_incremental_saver_roundtrip(tmp_path):
    output_path = tmp_path / "embeddings.parquet"
    saver = IncrementalSaver(str(output_path), chunk_size=2)

    results = [np.arange(3), np.arange(3)]
    saver.save_checkpoint(
        processed_count=2,
        total_count=4,
        results=results,
        start_index=0,
    )

    checkpoint = Path(str(output_path).replace(".parquet", "_checkpoint.json"))
    assert checkpoint.exists()

    state = saver.load_checkpoint()
    assert state.processed_count == 2

    embeddings_dir = tmp_path / "embeddings"
    files = list(embeddings_dir.iterdir())
    assert files
    part_file = files[0]
    if part_file.suffix == ".parquet":
        df = pd.read_parquet(part_file)
        assert len(df) == 2
    else:
        data = json.loads(part_file.read_text(encoding="utf-8"))
        assert len(data) == 2

    saver.cleanup()
    assert not checkpoint.exists()
