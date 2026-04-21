"""Tests for the chunked-schema marker on the Chroma collection.

After chunking ships, existing collections have one-vector-per-paper
records. To guide users to rebuild, ZoteroSemanticSearch checks whether
the collection carries a ``chunked: True`` marker and — when it's
missing on a non-empty collection — logs a one-shot warning telling the
user to run ``zotero-mcp update-db --force-rebuild``.

When the user performs a full rebuild, the marker is set so the warning
stops firing.
"""
from __future__ import annotations

import logging
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class _FakeCollection:
    def __init__(self, count: int = 0, metadata: dict | None = None):
        self._count = count
        self.metadata = metadata if metadata is not None else None
        self.modify_calls: list[dict] = []

    def count(self):
        return self._count

    def get(self, where=None, ids=None, limit=None, include=None):
        return {"ids": [], "metadatas": []}

    def delete(self, where=None, ids=None):
        pass

    def modify(self, metadata=None, name=None):
        if metadata is not None:
            self.modify_calls.append(dict(metadata))
            # Merge semantics matches how our code expects to use it.
            if self.metadata is None:
                self.metadata = {}
            self.metadata.update(metadata)


class _FakeChroma:
    def __init__(self, count: int = 0, metadata: dict | None = None):
        self.collection = _FakeCollection(count=count, metadata=metadata)
        self.embedding_max_tokens = 8000
        self.reset_calls = 0

    def reset_collection(self):
        self.reset_calls += 1
        # After reset, metadata is blank on the fresh collection.
        self.collection = _FakeCollection(count=0, metadata=None)

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def truncate_text(self, text, max_tokens=None):
        return text

    def delete_documents_where(self, where):
        pass

    def get_existing_ids(self, ids):
        return set()


def _make_search(monkeypatch, chroma):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    return semantic_search.ZoteroSemanticSearch(chroma_client=chroma)


# ---------------------------------------------------------------------------
# Warning behavior
# ---------------------------------------------------------------------------


def test_fresh_empty_collection_sets_marker_no_warning(monkeypatch, caplog):
    chroma = _FakeChroma(count=0, metadata=None)
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        _make_search(monkeypatch, chroma)

    # No legacy-index warning on empty collection.
    warnings = [r for r in caplog.records if "Legacy" in r.getMessage() or "--force-rebuild" in r.getMessage()]
    assert warnings == []
    # Marker was set.
    assert chroma.collection.metadata.get("chunked") is True


def test_non_empty_collection_without_marker_warns_once(monkeypatch, caplog):
    chroma = _FakeChroma(count=100, metadata=None)
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        _make_search(monkeypatch, chroma)

    warnings = [
        r for r in caplog.records
        if "force-rebuild" in r.getMessage().lower() or "chunk" in r.getMessage().lower()
    ]
    assert len(warnings) == 1, f"expected exactly one warning, got {[w.getMessage() for w in warnings]}"

    # Marker NOT set — collection still has legacy records; the user must rebuild.
    assert chroma.collection.metadata is None or "chunked" not in (chroma.collection.metadata or {})


def test_marker_present_suppresses_warning(monkeypatch, caplog):
    chroma = _FakeChroma(count=100, metadata={"chunked": True, "chunk_version": 1})
    with caplog.at_level(logging.WARNING, logger="zotero_mcp.semantic_search"):
        _make_search(monkeypatch, chroma)

    warnings = [
        r for r in caplog.records
        if "force-rebuild" in r.getMessage().lower() or "legacy" in r.getMessage().lower()
    ]
    assert warnings == []


def test_force_full_rebuild_sets_marker_after_reset(monkeypatch):
    chroma = _FakeChroma(count=100, metadata=None)
    s = _make_search(monkeypatch, chroma)

    # Short-circuit item fetch.
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch,
        "_get_items_from_source",
        lambda self, **kw: [],
    )
    s.update_database(force_full_rebuild=True)

    # Reset was called …
    assert chroma.reset_calls == 1
    # … and the fresh collection now carries the chunked marker.
    assert chroma.collection.metadata.get("chunked") is True
    assert chroma.collection.metadata.get("chunk_version") == 1
