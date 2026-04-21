"""Tests for ZoteroSemanticSearch.delete_item becoming chunk-aware.

Before chunking, delete_item(key) removed the bare ``key`` document.
After chunking, a paper has N+1 records with IDs ``key#-1`` … ``key#N``,
so deletion must happen via metadata match (``item_key == key``) to
remove all of them atomically.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class _FakeCollection:
    def __init__(self):
        self.data: dict[str, tuple[str, dict]] = {}

    def get(self, where=None, ids=None, limit=None, include=None):
        if where:
            hits = [
                (k, v) for k, v in self.data.items()
                if all(v[1].get(wk) == wv for wk, wv in where.items())
            ]
            if limit:
                hits = hits[:limit]
            return {"ids": [k for k, _ in hits], "metadatas": [v[1] for _, v in hits]}
        return {"ids": [], "metadatas": []}

    def delete(self, where=None, ids=None):
        if where:
            victims = [
                k for k, v in self.data.items()
                if all(v[1].get(wk) == wv for wk, wv in where.items())
            ]
            for k in victims:
                del self.data[k]
        if ids:
            for i in ids:
                self.data.pop(i, None)


class _FakeChroma:
    def __init__(self):
        self.collection = _FakeCollection()
        self.embedding_max_tokens = 8000
        self.delete_ids_calls: list[list[str]] = []
        self.delete_where_calls: list[dict] = []

    def upsert_documents(self, documents, metadatas, ids):
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            self.collection.data[doc_id] = (doc, meta)

    def delete_documents(self, ids):
        self.delete_ids_calls.append(list(ids))
        self.collection.delete(ids=ids)

    def delete_documents_where(self, where):
        self.delete_where_calls.append(dict(where))
        self.collection.delete(where=where)

    def truncate_text(self, text, max_tokens=None):
        return text

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self.collection.data}


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    return semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())


def test_delete_item_removes_all_chunks(search):
    """A paper with multiple chunks should have all of them removed."""
    # Seed 4 chunks for one paper + 1 chunk for a different paper.
    client = search.chroma_client
    client.upsert_documents(
        documents=["s", "c0", "c1", "c2", "other"],
        metadatas=[
            {"item_key": "PAPERAAA", "chunk_idx": -1},
            {"item_key": "PAPERAAA", "chunk_idx": 0},
            {"item_key": "PAPERAAA", "chunk_idx": 1},
            {"item_key": "PAPERAAA", "chunk_idx": 2},
            {"item_key": "PAPERBBB", "chunk_idx": -1},
        ],
        ids=["PAPERAAA#-1", "PAPERAAA#0", "PAPERAAA#1", "PAPERAAA#2", "PAPERBBB#-1"],
    )

    ok = search.delete_item("PAPERAAA")
    assert ok is True

    # Every PAPERAAA chunk is gone; PAPERBBB is intact.
    remaining = list(client.collection.data.keys())
    assert "PAPERBBB#-1" in remaining
    assert all("PAPERAAA" not in k for k in remaining)


def test_delete_item_uses_metadata_filter_not_id_list(search):
    """The implementation must route via delete_documents_where."""
    client = search.chroma_client
    client.upsert_documents(
        documents=["s", "c0"],
        metadatas=[
            {"item_key": "PAPERCCC", "chunk_idx": -1},
            {"item_key": "PAPERCCC", "chunk_idx": 0},
        ],
        ids=["PAPERCCC#-1", "PAPERCCC#0"],
    )

    search.delete_item("PAPERCCC")

    assert client.delete_where_calls == [{"item_key": "PAPERCCC"}]


def test_delete_item_removes_legacy_bare_id(search):
    """A legacy record stored with id=<key> (no ``#``) should also be removed,
    since its metadata still carries item_key=<key>."""
    client = search.chroma_client
    client.upsert_documents(
        documents=["legacy"],
        metadatas=[{"item_key": "PAPERDDD"}],
        ids=["PAPERDDD"],  # legacy bare id
    )
    search.delete_item("PAPERDDD")
    assert "PAPERDDD" not in client.collection.data


def test_delete_item_returns_false_on_backend_error(search, monkeypatch):
    def _boom(where):
        raise RuntimeError("simulated backend failure")
    monkeypatch.setattr(search.chroma_client, "delete_documents_where", _boom)

    assert search.delete_item("ANY_KEY") is False
