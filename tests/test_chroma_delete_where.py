"""Tests for ChromaClient.delete_documents_where.

Exercises the real ChromaClient against a temp persist directory so we
verify the delete(where=...) wiring, not a hand-rolled fake.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


def _client(tmp_path, name="test_delete_where"):
    from zotero_mcp.chroma_client import ChromaClient
    return ChromaClient(
        collection_name=name,
        persist_directory=str(tmp_path),
        embedding_model="default",
    )


def _seed(client):
    # Attach pre-computed embeddings so we skip the default EF model load
    # on first call and keep the test fast.
    client.collection.add(
        ids=["1", "2", "3", "4"],
        documents=["doc one", "doc two", "doc three", "doc four"],
        metadatas=[
            {"group": "a", "item_key": "PAPERAAA"},
            {"group": "a", "item_key": "PAPERBBB"},
            {"group": "b", "item_key": "PAPERCCC"},
            {"group": "b", "item_key": "PAPERCCC"},
        ],
        embeddings=[
            [0.1, 0.2, 0.3],
            [0.2, 0.3, 0.4],
            [0.3, 0.4, 0.5],
            [0.4, 0.5, 0.6],
        ],
    )


def test_delete_documents_where_removes_matching_group(tmp_path):
    client = _client(tmp_path)
    _seed(client)
    assert client.collection.count() == 4

    client.delete_documents_where({"group": "a"})

    assert client.collection.count() == 2
    remaining = client.collection.get()
    assert sorted(remaining["ids"]) == ["3", "4"]
    for meta in remaining["metadatas"]:
        assert meta["group"] == "b"


def test_delete_documents_where_by_item_key(tmp_path):
    client = _client(tmp_path, name="test_delete_by_key")
    _seed(client)

    client.delete_documents_where({"item_key": "PAPERCCC"})

    assert client.collection.count() == 2
    remaining = client.collection.get()
    for meta in remaining["metadatas"]:
        assert meta["item_key"] != "PAPERCCC"


def test_delete_documents_where_no_match_is_noop(tmp_path):
    client = _client(tmp_path, name="test_delete_no_match")
    _seed(client)

    client.delete_documents_where({"group": "nonexistent"})

    assert client.collection.count() == 4


def test_has_documents_where_true_when_match_exists(tmp_path):
    client = _client(tmp_path, name="test_has_docs_true")
    _seed(client)
    assert client.has_documents_where({"group": "a"}) is True


def test_has_documents_where_false_when_no_match(tmp_path):
    client = _client(tmp_path, name="test_has_docs_false")
    _seed(client)
    assert client.has_documents_where({"group": "nonexistent"}) is False


def test_has_documents_where_on_empty_collection(tmp_path):
    client = _client(tmp_path, name="test_has_docs_empty")
    assert client.has_documents_where({"group": "a"}) is False


def test_delete_documents_where_raises_on_backend_error(tmp_path, monkeypatch):
    """Backend errors should propagate, matching delete_documents behavior."""
    client = _client(tmp_path, name="test_delete_err")
    _seed(client)

    def _boom(*_a, **_k):
        raise RuntimeError("simulated backend failure")

    monkeypatch.setattr(client.collection, "delete", _boom)
    with pytest.raises(RuntimeError, match="simulated backend failure"):
        client.delete_documents_where({"group": "a"})
