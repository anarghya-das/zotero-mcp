"""Token-window chunker for passage-level semantic retrieval.

Pure helper module. Intentionally free of Zotero/Chroma imports so it can
be unit-tested without project dependencies.
"""
from __future__ import annotations

import re
from typing import Any

_WHITESPACE_RE = re.compile(r"\s+")
_CHAR_TO_TOKEN_RATIO = 4  # rough char-per-token estimate for English-heavy text


def _default_tokenizer() -> Any | None:
    """Return a tiktoken cl100k_base encoder, or None if tiktoken is missing.

    Separated so tests can monkeypatch to simulate a tiktoken-free env.
    """
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        return None


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_fulltext(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
    tokenizer: Any = None,
) -> list[str]:
    """Split text into overlapping token windows suitable for embedding.

    Empty/whitespace-only input returns []. Text that fits in a single
    window returns a one-element list. Otherwise emits sliding windows of
    ``target_tokens`` with ``overlap_tokens`` shared tokens between
    successive windows.

    If ``tokenizer`` is None, tries tiktoken cl100k_base. When tiktoken is
    unavailable, falls back to a character-based approximation (~4 chars
    per token).
    """
    if target_tokens <= overlap_tokens:
        raise ValueError("target_tokens must be greater than overlap_tokens")

    text = _normalize_whitespace(text)
    if not text:
        return []

    if tokenizer is None:
        tokenizer = _default_tokenizer()

    if tokenizer is not None:
        tokens = tokenizer.encode(text, disallowed_special=())
        if len(tokens) <= target_tokens:
            return [text]
        chunks: list[str] = []
        step = target_tokens - overlap_tokens
        start = 0
        while start < len(tokens):
            window = tokens[start : start + target_tokens]
            chunks.append(tokenizer.decode(window))
            if start + target_tokens >= len(tokens):
                break
            start += step
        return chunks

    # Char-based fallback when tiktoken is unavailable.
    target_chars = target_tokens * _CHAR_TO_TOKEN_RATIO
    overlap_chars = overlap_tokens * _CHAR_TO_TOKEN_RATIO
    if len(text) <= target_chars:
        return [text]
    chunks = []
    step = target_chars - overlap_chars
    start = 0
    while start < len(text):
        chunks.append(text[start : start + target_chars])
        if start + target_chars >= len(text):
            break
        start += step
    return chunks
