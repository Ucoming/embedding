import numpy as np
import pandas as pd

from tool_embedding.models.base import EmbeddingBackend
from tool_embedding.pipeline import EmbeddingProcessor, ProcessorConfig
from tool_embedding.storage import IncrementalSaver


class DummyBackend(EmbeddingBackend):
    def __init__(self, max_tokens: int = 5):
        self.name = "dummy"
        self.max_tokens = max_tokens
        self.calls = []

    def count_tokens(self, text: str) -> int:
        return len(text)

    def embed_batch(self, texts, *, batch_size: int = 1):
        self.calls.append(tuple(texts))
        return [np.arange(len(text), dtype=float) for text in texts]

    def supports_batching(self) -> bool:
        return True


def test_processor_handles_long_text_and_checkpoint(tmp_path):
    backend = DummyBackend(max_tokens=5)
    processor = EmbeddingProcessor(
        bge_backend=backend,
        config=ProcessorConfig(chunk_overlap=0, checkpoint_chunk_size=1),
    )

    df = pd.DataFrame({"content": ["abc", "abcdefg"]})
    output_path = tmp_path / "results.parquet"

    result = processor.process_dataframe(
        df,
        text_column="content",
        use_bge=True,
        batch_size=2,
        output_path=str(output_path),
        checkpoint_interval=1,
    )

    assert "bge_embedding" in result.columns
    first, second = result["bge_embedding"].tolist()
    assert isinstance(first, list)
    assert isinstance(second, list)
    assert len(second) > 1  # long text was chunked

    # Simulate a resume by keeping the checkpoint and part file.
    saver = IncrementalSaver(str(output_path) + "_bge", chunk_size=1)
    saver.save_checkpoint(
        processed_count=1,
        total_count=2,
        results=[np.arange(3), None],
        start_index=0,
    )

    calls_before_resume = len(backend.calls)

    resumed = processor.process_dataframe(
        df,
        text_column="content",
        use_bge=True,
        batch_size=2,
        output_path=str(output_path),
        checkpoint_interval=1,
    )

    assert resumed["bge_embedding"].iloc[0] is not None
    assert len(backend.calls) > calls_before_resume
