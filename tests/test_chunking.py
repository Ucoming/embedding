from tool_embedding.chunking import heuristic_token_counter, split_text_by_tokens


def test_split_text_respects_max_tokens():
    text = "这是一段非常长的文本。" * 5
    chunks = split_text_by_tokens(text, max_tokens=10)
    assert chunks, "Expected chunks to be generated"
    for chunk in chunks:
        assert heuristic_token_counter(chunk) <= 10


def test_split_text_with_overlap():
    text = "句子1。句子2。句子3。句子4。"
    chunks = split_text_by_tokens(text, max_tokens=5, chunk_overlap=2)
    assert len(chunks) > 1
    # Overlap is approximate but ensure neighbouring chunks share content.
    for left, right in zip(chunks, chunks[1:]):
        assert any(char in right for char in left[-3:])
