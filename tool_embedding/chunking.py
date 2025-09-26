"""Utilities for counting tokens and chunking text."""
from __future__ import annotations

from typing import Callable, List, Optional
import re

TokenCounter = Callable[[str], int]


def heuristic_token_counter(text: str) -> int:
    """Estimate token counts when a tokenizer is unavailable.

    The heuristic favours slightly over-estimating to avoid exceeding the
    provider limits. Chinese characters are counted roughly as one token,
    latin characters as 0.3 token and the remaining characters as 0.5.
    """

    if not text:
        return 0

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    english_chars = len(re.findall(r"[a-zA-Z]", text))
    other_chars = max(len(text) - chinese_chars - english_chars, 0)
    estimated_tokens = chinese_chars * 1.2 + english_chars * 0.3 + other_chars * 0.5
    return int(estimated_tokens)


def split_text_by_tokens(
    text: str,
    max_tokens: int,
    *,
    counter: Optional[TokenCounter] = None,
    sentence_delimiters: str = r"[。！？；\n]",
    chunk_overlap: int = 0,
) -> List[str]:
    """Split *text* into chunks that stay below ``max_tokens``.

    Args:
        text: The string to chunk.
        max_tokens: Maximum token length for a chunk.
        counter: Optional callable returning the number of tokens for a chunk.
            When omitted ``heuristic_token_counter`` is used.
        sentence_delimiters: Regex used to split the text into candidate
            sentences.
        chunk_overlap: Number of tokens of overlap to maintain between
            neighbouring chunks. The overlap is approximate when a heuristic
            counter is used.

    Returns:
        A list of chunks in their original order.
    """

    if not text or not text.strip():
        return []

    counter = counter or heuristic_token_counter
    if counter(text) <= max_tokens:
        return [text]

    sentences = [
        sentence.strip()
        for sentence in re.split(sentence_delimiters, text)
        if sentence.strip()
    ]

    chunks: List[str] = []
    current_chunk: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sentence_tokens = counter(sentence)

        if sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append("".join(current_chunk))
                current_chunk = []
                current_tokens = 0

            # Approximate how many characters we can keep per chunk.
            chars_per_token = len(sentence) / max(sentence_tokens, 1)
            max_chars = max(int(max_tokens * chars_per_token * 0.8), 1)

            for i in range(0, len(sentence), max_chars):
                chunk = sentence[i : i + max_chars]
                if chunk.strip():
                    chunks.append(chunk)
            continue

        # If adding the sentence would overflow the chunk we finalise the
        # current chunk and start a new one.
        if current_tokens + sentence_tokens > max_tokens:
            if current_chunk:
                chunks.append("".join(current_chunk))

            if chunk_overlap > 0 and chunks:
                # Keep an approximate overlap by reusing the tail of the
                # previous chunk.
                overlap_tokens = 0
                overlap_sentences: List[str] = []
                for prev_sentence in reversed(current_chunk):
                    overlap_tokens += counter(prev_sentence)
                    overlap_sentences.insert(0, prev_sentence)
                    if overlap_tokens >= chunk_overlap:
                        break
                current_chunk = overlap_sentences[:]
                current_tokens = sum(counter(s) for s in current_chunk)
            else:
                current_chunk = []
                current_tokens = 0

        current_chunk.append(sentence)
        current_tokens += sentence_tokens

    if current_chunk:
        chunks.append("".join(current_chunk))

    return chunks
