"""Tests that ChromaClient refuses to silently wipe non-empty collections.

Background: on 2026-04-23 the user's 1,652-chunk collection was silently
destroyed when a diagnostic CLI command spawned a fresh ChromaClient and
ChromaDB raised "embedding function conflict" between the stored spec
and the custom OpenAIEmbeddingFunction. The prior exception handler
auto-called delete_collection + create_collection, destroying the index
without warning or confirmation.

Policy:
- If the collection has > 0 documents and an EF/model-drift is detected,
  RAISE a clear error explaining how to rebuild explicitly. Never silently
  drop.
- If the collection is empty (count == 0), auto-drop is safe — proceed.
- The env var ZOTERO_MCP_ALLOW_AUTO_RESET=1 overrides the guard for CI
  and intentional rebuilds (escape hatch, documented).
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


def _client(tmp_path, name="guard_test"):
    from zotero_mcp.chroma_client import ChromaClient
    return ChromaClient(
        collection_name=name,
        persist_directory=str(tmp_path),
        embedding_model="default",
    )


# ---------------------------------------------------------------------------
# Decision helper — pure unit test
# ---------------------------------------------------------------------------


def test_auto_drop_allowed_when_collection_empty(tmp_path, monkeypatch):
    monkeypatch.delenv("ZOTERO_MCP_ALLOW_AUTO_RESET", raising=False)
    client = _client(tmp_path, name="allowed_empty")
    assert client._auto_drop_allowed(existing_count=0) is True


def test_auto_drop_refused_when_collection_has_docs(tmp_path, monkeypatch):
    monkeypatch.delenv("ZOTERO_MCP_ALLOW_AUTO_RESET", raising=False)
    client = _client(tmp_path, name="refused_with_docs")
    assert client._auto_drop_allowed(existing_count=1) is False
    assert client._auto_drop_allowed(existing_count=1652) is False


def test_env_override_forces_auto_drop_even_with_docs(tmp_path, monkeypatch):
    monkeypatch.setenv("ZOTERO_MCP_ALLOW_AUTO_RESET", "1")
    client = _client(tmp_path, name="env_override")
    assert client._auto_drop_allowed(existing_count=1000) is True


# ---------------------------------------------------------------------------
# Refusal helper — raises with useful rescue info
# ---------------------------------------------------------------------------


def test_refusal_raises_with_rescue_commands(tmp_path):
    client = _client(tmp_path, name="refusal_msg")
    with pytest.raises(RuntimeError) as exc:
        client._raise_auto_drop_refused(
            reason="test reason", existing_count=108
        )
    msg = str(exc.value)
    # Must include the count so the user knows what's at stake.
    assert "108" in msg
    # Must name the override env var.
    assert "ZOTERO_MCP_ALLOW_AUTO_RESET" in msg
    # Must mention --force-rebuild as the explicit opt-in path.
    assert "--force-rebuild" in msg
    # Must include the persist directory path so user can back it up.
    assert client.persist_directory in msg


# ---------------------------------------------------------------------------
# Integration: simulate an EF conflict on a populated collection.
# The destructive branch must NOT call delete_collection — must raise.
# ---------------------------------------------------------------------------


def test_populated_collection_survives_ef_conflict(tmp_path, monkeypatch):
    """Seed a collection, then simulate the EF-conflict exception branch
    re-firing on a fresh ChromaClient init. Assert no delete_collection call."""
    from zotero_mcp.chroma_client import ChromaClient
    from chromadb.api.client import Client as ChromaConcreteClient

    # Build and seed
    client = ChromaClient(
        collection_name="survive_conflict",
        persist_directory=str(tmp_path),
        embedding_model="default",
    )
    client.collection.add(
        ids=["a", "b", "c"],
        documents=["doc a", "doc b", "doc c"],
        embeddings=[[0.1, 0.2, 0.3], [0.2, 0.3, 0.4], [0.3, 0.4, 0.5]],
    )
    assert client.collection.count() == 3

    monkeypatch.delenv("ZOTERO_MCP_ALLOW_AUTO_RESET", raising=False)

    # Force get_or_create_collection to raise the EF-conflict error on
    # the next init. Capture delete_collection to ensure it's never called.
    delete_called = {"count": 0}

    orig_delete = ChromaConcreteClient.delete_collection

    def fake_get_or_create(self, *a, **kw):
        raise RuntimeError("embedding function conflict: stored != provided")

    def tracked_delete(self, *a, **kw):
        delete_called["count"] += 1
        return orig_delete(self, *a, **kw)

    monkeypatch.setattr(
        ChromaConcreteClient, "get_or_create_collection", fake_get_or_create
    )
    monkeypatch.setattr(
        ChromaConcreteClient, "delete_collection", tracked_delete
    )

    # Attempt to re-instantiate — must raise, must NOT call delete_collection.
    with pytest.raises(RuntimeError) as exc:
        ChromaClient(
            collection_name="survive_conflict",
            persist_directory=str(tmp_path),
            embedding_model="default",
        )
    msg = str(exc.value)
    # Must be our refusal, not the raw conflict — signals the guard kicked in.
    assert "Refusing" in msg or "refusing" in msg
    assert "--force-rebuild" in msg
    assert delete_called["count"] == 0


def test_empty_collection_ef_conflict_is_auto_resolved(tmp_path, monkeypatch):
    """When no data is at stake, the auto-drop path should still work — we
    only want to guard non-empty collections."""
    from zotero_mcp.chroma_client import ChromaClient
    from chromadb.api.client import Client as ChromaConcreteClient

    # Build an empty collection
    ChromaClient(
        collection_name="empty_conflict",
        persist_directory=str(tmp_path),
        embedding_model="default",
    )

    monkeypatch.delenv("ZOTERO_MCP_ALLOW_AUTO_RESET", raising=False)

    # Re-instantiate with a forced EF conflict. On an empty collection the
    # guard should permit the auto-drop and recreate.
    raise_once = {"done": False}
    orig_get_or_create = ChromaConcreteClient.get_or_create_collection

    def fake_get_or_create(self, *a, **kw):
        if not raise_once["done"]:
            raise_once["done"] = True
            raise RuntimeError("embedding function conflict")
        return orig_get_or_create(self, *a, **kw)

    monkeypatch.setattr(
        ChromaConcreteClient, "get_or_create_collection", fake_get_or_create
    )

    # Should not raise — the collection was empty, safe to auto-drop.
    client = ChromaClient(
        collection_name="empty_conflict",
        persist_directory=str(tmp_path),
        embedding_model="default",
    )
    assert client.collection.count() == 0
