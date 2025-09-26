import json
from pathlib import Path

import numpy as np
import pandas as pd

from tool_embedding.storage import IncrementalSaver, iter_part_paths


def test_incremental_saver_roundtrip(tmp_path):
    output_path = tmp_path / "embeddings.parquet"
    saver = IncrementalSaver(str(output_path), chunk_size=2, parts_per_directory=1)

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
    files = list(embeddings_dir.rglob("part-*"))
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


def test_iter_part_paths_nested(tmp_path):
    base_dir = tmp_path / "embeddings"
    group0 = base_dir / "group-00000"
    group1 = base_dir / "group-00001"
    group0.mkdir(parents=True)
    group1.mkdir(parents=True)

    (group0 / "part-00001.json").write_text("[]", encoding="utf-8")
    (group1 / "part-00002.parquet").write_text("", encoding="utf-8")
    (group0 / "part-00000.parquet").write_text("", encoding="utf-8")

    files = list(iter_part_paths(str(base_dir)))
    assert [Path(path).name for path in files] == [
        "part-00000.parquet",
        "part-00001.json",
        "part-00002.parquet",
    ]
