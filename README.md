# embedding

This repository provides a production-ready toolkit for generating text embeddings using both local open-source models and remotely hosted API providers. The refactored codebase focuses on performance, resumable processing for very large datasets, and a clean Python package layout.

## Key features

- Modular architecture with dedicated packages for model backends, chunking utilities, checkpointing and the high-level processing pipeline.
- Pluggable embedding backends with built-in support for SentenceTransformer (e.g. BGE/Qwen) and OpenAI-compatible APIs.
- Robust token-aware chunking with optional overlap to keep prompts within provider limits.
- Incremental Parquet part files and JSON checkpoints that allow safe restarts after interruptions, now automatically fanning out into sub-directories to keep enormous outputs manageable.
- Comprehensive unit tests covering chunking, checkpoint persistence and pipeline resume behaviour.

## Package structure

```
tool_embedding/
├── __init__.py           # Public API (processor factory, quick pipeline helper)
├── chunking.py           # Token estimation and chunking helpers
├── data.py               # Data loading utilities
├── models/               # Embedding backend implementations
│   ├── base.py
│   ├── bge.py
│   └── openai.py
├── pipeline.py           # EmbeddingProcessor orchestration logic
└── storage.py            # Incremental saving/checkpoint helpers
```

The legacy `tool_embedding.py` module now simply re-exports the package API for backwards compatibility.

## Usage

```python
from tool_embedding import quick_embedding_pipeline

result_df = quick_embedding_pipeline(
    data_path="data/messages.parquet",
    output_path="output/messages_embeddings.parquet",
    text_column="content",
    use_bge=True,
    use_gpt=False,
    bge_model_name="Qwen/Qwen3-Embedding-8B",
    bge_model_location="Qwen3-Embedding-8B",  # optional default local directory
    batch_size=128,
    checkpoint_interval=5000,
)
```

For more control instantiate the processor manually:

```python
from tool_embedding import create_processor

processor = create_processor(
    load_bge=True,
    load_gpt=False,
    bge_model_name="Qwen/Qwen3-Embedding-8B",
    bge_model_location="Qwen3-Embedding-8B",  # override the default local cache path
)
df = ...  # pandas DataFrame
embeddings = processor.process_dataframe(df, text_column="content", output_path="output.parquet")
```

## Development

Install the project requirements and run the tests:

```bash
pip install -r requirements.txt  # if available
pytest
```

The tests rely on lightweight dummy backends so no heavyweight models or API keys are required.
