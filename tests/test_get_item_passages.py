"""Tests for the zotero_get_item_passages MCP tool.

Two modes:
- query=None → return every chunk for the paper, ordered by chunk_idx
  ascending (summary `#-1` first when present), capped at `limit`.
- query set → top-k chunks most relevant to `query`, filtered by
  metadata to just this item_key.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


class _FakeCollection:
    """Minimal in-memory Chroma collection."""

    def __init__(self, items: list[tuple[str, str, dict]]):
        # items: list of (id, doc, metadata)
        self._items = items

    def get(self, where=None, ids=None, include=None, limit=None):
        hits = [
            (i, d, m) for (i, d, m) in self._items
            if (not where) or all(m.get(wk) == wv for wk, wv in where.items())
        ]
        if limit:
            hits = hits[:limit]
        return {
            "ids": [i for (i, _, _) in hits],
            "documents": [d for (_, d, _) in hits],
            "metadatas": [m for (_, _, m) in hits],
        }


class _FakeChromaClient:
    """Wraps a _FakeCollection and mimics ChromaClient.search."""

    def __init__(self, items):
        self.collection = _FakeCollection(items)
        self._items = items

    def search(self, query_texts=None, n_results=10, where=None, where_document=None):
        # Naive "search": filter by `where`, sort items so that text containing
        # the query string scores highest. Good enough for unit tests.
        q = (query_texts or [""])[0].lower()
        hits = [
            (i, d, m) for (i, d, m) in self._items
            if (not where) or all(m.get(wk) == wv for wk, wv in where.items())
        ]
        def _score(triple):
            _, d, _ = triple
            # Higher is better; negate for sort-asc convention on distances.
            return -d.lower().count(q)

        hits.sort(key=_score)
        hits = hits[:n_results]
        return {
            "ids": [[i for (i, _, _) in hits]],
            "distances": [[0.1 * (k + 1) for k in range(len(hits))]],
            "documents": [[d for (_, d, _) in hits]],
            "metadatas": [[m for (_, _, m) in hits]],
        }


class _FakeSearch:
    """Stand-in for ZoteroSemanticSearch with just the bits we need."""

    def __init__(self, items):
        self.chroma_client = _FakeChromaClient(items)


def _install(monkeypatch, items):
    import zotero_mcp.semantic_search as ss
    fake = _FakeSearch(items)
    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: fake)
    return fake


def _tool():
    from zotero_mcp.tools.retrieval import get_item_passages
    return get_item_passages


# ---------------------------------------------------------------------------
# query=None → chronological, summary first, capped
# ---------------------------------------------------------------------------


def test_all_chunks_ordered_by_chunk_idx(monkeypatch, dummy_ctx):
    items = [
        ("K1#2", "chunk two", {"item_key": "K1", "chunk_idx": 2, "title": "Paper"}),
        ("K1#-1", "summary", {"item_key": "K1", "chunk_idx": -1, "title": "Paper"}),
        ("K1#0", "chunk zero", {"item_key": "K1", "chunk_idx": 0, "title": "Paper"}),
        ("K1#1", "chunk one", {"item_key": "K1", "chunk_idx": 1, "title": "Paper"}),
        # Unrelated paper — must not appear
        ("K2#-1", "other", {"item_key": "K2", "chunk_idx": -1, "title": "Other"}),
    ]
    _install(monkeypatch, items)

    out = _tool()(item_key="K1", query=None, limit=10, ctx=dummy_ctx)

    # Summary appears once, before fulltext chunks.
    assert out.count("Summary") == 1
    assert out.index("Summary") < out.index("Passage 0")
    assert out.index("Passage 0") < out.index("Passage 1")
    assert out.index("Passage 1") < out.index("Passage 2")
    # Content sanity
    assert "summary" in out
    assert "chunk zero" in out and "chunk one" in out and "chunk two" in out
    # Unrelated paper's chunk is NOT present
    assert "other" not in out.lower() or "Other" not in out


def test_limit_caps_output(monkeypatch, dummy_ctx):
    items = [
        ("K3#-1", "summary", {"item_key": "K3", "chunk_idx": -1, "title": "P"}),
        ("K3#0", "c0", {"item_key": "K3", "chunk_idx": 0, "title": "P"}),
        ("K3#1", "c1", {"item_key": "K3", "chunk_idx": 1, "title": "P"}),
        ("K3#2", "c2", {"item_key": "K3", "chunk_idx": 2, "title": "P"}),
    ]
    _install(monkeypatch, items)

    out = _tool()(item_key="K3", query=None, limit=2, ctx=dummy_ctx)

    # Only first 2 after ordering: summary + chunk 0
    assert "summary" in out
    assert "c0" in out
    assert "c1" not in out
    assert "c2" not in out


def test_unknown_item_returns_helpful_message(monkeypatch, dummy_ctx):
    _install(monkeypatch, [])
    out = _tool()(item_key="NOPE1234", query=None, ctx=dummy_ctx)
    assert "NOPE1234" in out
    low = out.lower()
    assert "update-db" in low or "no passages" in low or "not found" in low


# ---------------------------------------------------------------------------
# query set → similarity-ordered, filtered to item_key
# ---------------------------------------------------------------------------


def test_query_returns_ranked_passages_for_item(monkeypatch, dummy_ctx):
    items = [
        ("K4#-1", "Some paper about cardiac monitoring.",
         {"item_key": "K4", "chunk_idx": -1, "title": "Cardio paper"}),
        ("K4#0", "photoplethysmography signal quality matters",
         {"item_key": "K4", "chunk_idx": 0, "title": "Cardio paper"}),
        ("K4#1", "unrelated text about kittens",
         {"item_key": "K4", "chunk_idx": 1, "title": "Cardio paper"}),
        # Unrelated paper with the same query terms — must be filtered out
        ("K5#0", "photoplethysmography in other paper",
         {"item_key": "K5", "chunk_idx": 0, "title": "Other"}),
    ]
    _install(monkeypatch, items)

    out = _tool()(item_key="K4", query="photoplethysmography", limit=5, ctx=dummy_ctx)

    assert "photoplethysmography signal quality matters" in out
    # Other paper's content must NOT appear
    assert "in other paper" not in out
    # Score label appears
    assert "score" in out.lower()


def test_query_result_is_markdown_with_passage_headers(monkeypatch, dummy_ctx):
    items = [
        ("K6#0", "first passage", {"item_key": "K6", "chunk_idx": 0, "title": "P"}),
        ("K6#1", "second passage", {"item_key": "K6", "chunk_idx": 1, "title": "P"}),
    ]
    _install(monkeypatch, items)

    out = _tool()(item_key="K6", query="first", ctx=dummy_ctx)
    assert "### " in out  # Some H3 header per passage


def test_respects_limit_on_query_mode(monkeypatch, dummy_ctx):
    items = [
        (f"K7#{i}", f"passage {i} foo", {"item_key": "K7", "chunk_idx": i, "title": "P"})
        for i in range(5)
    ]
    _install(monkeypatch, items)

    out = _tool()(item_key="K7", query="foo", limit=2, ctx=dummy_ctx)
    # With limit=2, at most 2 passage headings
    hdr_count = out.count("### Passage") + out.count("### Summary")
    assert hdr_count == 2
