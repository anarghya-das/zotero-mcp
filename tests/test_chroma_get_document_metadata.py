"""Tests for ChromaClient.get_document_metadata under the chunked schema.

Callers (semantic_search._process_item_batch) pass a bare Zotero item
key. Under the chunked schema, docs live at `key#-1`, `key#0`, ...
so the method must resolve a bare key to the summary chunk (`key#-1`).
A caller passing a full chunk ID should still work for backwards
compatibility.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


def _client(tmp_path, name="test_get_meta"):
    from zotero_mcp.chroma_client import ChromaClient
    return ChromaClient(
        collection_name=name,
        persist_directory=str(tmp_path),
        embedding_model="default",
    )


def _seed_chunked(client):
    """Seed two papers, each with summary + fulltext chunks."""
    client.collection.add(
        ids=["PAPERAAA#-1", "PAPERAAA#0", "PAPERAAA#1", "PAPERBBB#-1"],
        documents=["A summary", "A passage 0", "A passage 1", "B summary"],
        metadatas=[
            {"item_key": "PAPERAAA", "chunk_idx": -1, "has_fulltext": True, "date_modified": "2026-01-01"},
            {"item_key": "PAPERAAA", "chunk_idx": 0,  "has_fulltext": True, "date_modified": "2026-01-01"},
            {"item_key": "PAPERAAA", "chunk_idx": 1,  "has_fulltext": True, "date_modified": "2026-01-01"},
            {"item_key": "PAPERBBB", "chunk_idx": -1, "has_fulltext": False, "date_modified": "2026-01-02"},
        ],
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ],
    )


def test_resolves_bare_item_key_to_summary_chunk(tmp_path):
    client = _client(tmp_path, name="test_bare_key")
    _seed_chunked(client)

    meta = client.get_document_metadata("PAPERAAA")

    assert meta is not None
    assert meta["chunk_idx"] == -1
    assert meta["item_key"] == "PAPERAAA"
    assert meta["has_fulltext"] is True
    assert meta["date_modified"] == "2026-01-01"


def test_returns_summary_for_paper_without_fulltext(tmp_path):
    """A paper with no PDF has only the summary chunk; lookup must still find it."""
    client = _client(tmp_path, name="test_no_fulltext")
    _seed_chunked(client)

    meta = client.get_document_metadata("PAPERBBB")

    assert meta is not None
    assert meta["chunk_idx"] == -1
    assert meta["has_fulltext"] is False


def test_returns_none_for_unknown_key(tmp_path):
    client = _client(tmp_path, name="test_unknown")
    _seed_chunked(client)

    assert client.get_document_metadata("MISSINGX") is None


def test_accepts_full_chunk_id_for_backcompat(tmp_path):
    """If a caller already passes a full chunk ID, it should still work."""
    client = _client(tmp_path, name="test_full_id")
    _seed_chunked(client)

    meta = client.get_document_metadata("PAPERAAA#0")

    assert meta is not None
    assert meta["chunk_idx"] == 0
