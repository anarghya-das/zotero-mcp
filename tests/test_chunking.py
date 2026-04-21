"""Tests for src/zotero_mcp/chunking.py — token-window fulltext chunker."""
from __future__ import annotations

import math

import pytest

from zotero_mcp.chunking import chunk_fulltext


def _enc():
    import tiktoken
    return tiktoken.get_encoding("cl100k_base")


def test_empty_text_returns_empty_list():
    assert chunk_fulltext("", target_tokens=100, overlap_tokens=10) == []


def test_whitespace_only_returns_empty_list():
    assert chunk_fulltext("   \n\n\t  ", target_tokens=100, overlap_tokens=10) == []


def test_short_text_single_chunk_whitespace_normalized():
    chunks = chunk_fulltext(
        "  hello   world  \n\n foo ", target_tokens=100, overlap_tokens=10
    )
    assert chunks == ["hello world foo"]


def test_short_text_fits_in_one_chunk():
    chunks = chunk_fulltext("hello world", target_tokens=100, overlap_tokens=10)
    assert chunks == ["hello world"]


def test_long_text_chunk_count_matches_formula():
    enc = _enc()
    text = " ".join(str(i) for i in range(1, 1001))
    total = len(enc.encode(text, disallowed_special=()))

    target, overlap = 100, 20
    chunks = chunk_fulltext(text, target_tokens=target, overlap_tokens=overlap)

    expected = math.ceil((total - overlap) / (target - overlap))
    assert len(chunks) == expected, f"got {len(chunks)} chunks for {total} tokens"


def test_chunks_fit_within_target_tokens():
    enc = _enc()
    text = " ".join(str(i) for i in range(1, 1001))
    chunks = chunk_fulltext(text, target_tokens=100, overlap_tokens=20)
    for c in chunks:
        assert len(enc.encode(c, disallowed_special=())) <= 100


def test_consecutive_chunks_share_overlap():
    """Successive chunks should share content (the overlap window)."""
    text = " ".join(str(i) for i in range(1, 501))
    chunks = chunk_fulltext(text, target_tokens=100, overlap_tokens=20)
    assert len(chunks) >= 2

    for i in range(len(chunks) - 1):
        tail_words = set(chunks[i].split()[-15:])
        head_words = set(chunks[i + 1].split()[:25])
        shared = tail_words & head_words
        assert shared, f"no shared words between chunk {i} and {i+1}"


def test_raises_when_overlap_ge_target():
    with pytest.raises(ValueError):
        chunk_fulltext("foo bar", target_tokens=10, overlap_tokens=10)
    with pytest.raises(ValueError):
        chunk_fulltext("foo bar", target_tokens=5, overlap_tokens=10)


def test_fallback_when_tokenizer_unavailable(monkeypatch):
    """Char-based fallback when tiktoken is not importable.

    Monkeypatches the module-level _default_tokenizer() to simulate an
    environment without tiktoken. The char fallback uses ~4 chars/token.
    """
    from zotero_mcp import chunking
    monkeypatch.setattr(chunking, "_default_tokenizer", lambda: None)

    # 1000 chars, target=100 tokens ≈ 400 chars, overlap=10 tokens ≈ 40 chars.
    # step = 400 - 40 = 360.
    # start=0, 360, 720 → 3 windows (720+400=1120 >= 1000 terminates).
    text = "a" * 1000
    chunks = chunk_fulltext(text, target_tokens=100, overlap_tokens=10)

    assert len(chunks) == 3
    assert all(len(c) <= 400 for c in chunks)
    assert chunks[0][-40:] == chunks[1][:40]


def test_explicit_tokenizer_overrides_default(monkeypatch):
    """Passing an explicit tokenizer bypasses the default lookup."""
    from zotero_mcp import chunking

    # Make sure the default path would be broken — prove explicit wins.
    monkeypatch.setattr(chunking, "_default_tokenizer", lambda: None)

    enc = _enc()
    text = " ".join(str(i) for i in range(1, 501))
    chunks = chunk_fulltext(
        text, target_tokens=100, overlap_tokens=20, tokenizer=enc
    )
    # Should be multiple chunks (tokenized path), not the char fallback's count.
    assert len(chunks) >= 2
    for c in chunks:
        assert len(enc.encode(c, disallowed_special=())) <= 100
