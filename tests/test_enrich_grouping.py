"""Tests for _enrich_search_results grouping by item_key.

With chunking, multiple chunks from the same paper may appear in one
Chroma result set. The enrichment step should collapse them into one
result per paper (preserving first-appearance order), dedup the Zotero
fetch, and expose a passages list ordered by similarity.
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


class _RecordingZot:
    def __init__(self):
        self.calls: list[str] = []

    def item(self, key):
        self.calls.append(key)
        return {"key": key, "data": {"title": f"Title for {key}"}}


class _FakeChroma:
    embedding_max_tokens = 8000


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: _RecordingZot())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())
    return s


def _mock_results(ids_distances_docs):
    """Build a chroma_results dict from flat (id, distance, doc) triples."""
    ids = [[t[0] for t in ids_distances_docs]]
    distances = [[t[1] for t in ids_distances_docs]]
    documents = [[t[2] for t in ids_distances_docs]]
    metadatas = [[{"item_key": t[0].split("#")[0], "chunk_idx": (
        -1 if t[0].endswith("#-1") else int(t[0].split("#")[1])
    )} for t in ids_distances_docs]]
    return {
        "ids": ids,
        "distances": distances,
        "documents": documents,
        "metadatas": metadatas,
    }


def test_one_result_per_paper_preserves_first_appearance_order(search):
    # Mixed order: foo#0, bar#-1, foo#2, baz#0, bar#1
    results = _mock_results([
        ("foo#0", 0.10, "foo chunk 0"),
        ("bar#-1", 0.20, "bar summary"),
        ("foo#2", 0.30, "foo chunk 2"),
        ("baz#0", 0.40, "baz chunk 0"),
        ("bar#1", 0.50, "bar chunk 1"),
    ])
    enriched = search._enrich_search_results(results, query="q")
    item_keys = [r["item_key"] for r in enriched]
    assert item_keys == ["foo", "bar", "baz"]


def test_zotero_fetched_once_per_unique_item(search):
    results = _mock_results([
        ("foo#0", 0.1, "a"),
        ("foo#1", 0.2, "b"),
        ("foo#2", 0.3, "c"),
        ("bar#-1", 0.4, "d"),
    ])
    search._enrich_search_results(results, query="q")
    assert sorted(search.zotero_client.calls) == ["bar", "foo"]


def test_passages_populated_and_ordered_by_similarity(search):
    results = _mock_results([
        ("foo#0", 0.50, "low-rel chunk"),
        ("foo#2", 0.10, "high-rel chunk"),
        ("foo#1", 0.30, "mid-rel chunk"),
    ])
    enriched = search._enrich_search_results(results, query="q")
    assert len(enriched) == 1

    foo = enriched[0]
    assert foo["item_key"] == "foo"
    assert foo["matched_text"] == "high-rel chunk"
    # similarity_score = 1 - distance — best chunk has highest score.
    assert foo["similarity_score"] == pytest.approx(0.90)

    passages = foo["passages"]
    assert [p["chunk_idx"] for p in passages] == [2, 1, 0]  # sorted by score desc
    assert passages[0]["text"] == "high-rel chunk"
    assert passages[0]["similarity_score"] == pytest.approx(0.90)


def test_summary_chunk_exposed_in_passages(search):
    results = _mock_results([
        ("foo#-1", 0.25, "summary"),
        ("foo#0", 0.10, "fulltext chunk"),
    ])
    enriched = search._enrich_search_results(results, query="q")
    passages = enriched[0]["passages"]
    chunk_idxs = {p["chunk_idx"] for p in passages}
    assert -1 in chunk_idxs


def test_empty_results_returns_empty_list(search):
    assert search._enrich_search_results({"ids": [[]]}, query="q") == []
    assert search._enrich_search_results({}, query="q") == []


def test_legacy_bare_id_still_works(search):
    """A legacy record (no '#' in id) should still yield an enriched result."""
    results = {
        "ids": [["legacy_key"]],
        "distances": [[0.15]],
        "documents": [["legacy doc text"]],
        "metadatas": [[{"item_key": "legacy_key"}]],
    }
    enriched = search._enrich_search_results(results, query="q")
    assert len(enriched) == 1
    assert enriched[0]["item_key"] == "legacy_key"
    assert enriched[0]["matched_text"] == "legacy doc text"


def test_enrichment_failure_preserves_partial_result(search, monkeypatch):
    """If zotero_client.item(key) raises, we still return a result with error."""
    def _boom(key):
        raise RuntimeError("404")
    monkeypatch.setattr(search.zotero_client, "item", _boom)

    results = _mock_results([("foo#0", 0.1, "x")])
    enriched = search._enrich_search_results(results, query="q")
    assert len(enriched) == 1
    assert enriched[0]["item_key"] == "foo"
    assert "error" in enriched[0]


def test_passages_list_length_matches_chunks_for_paper(search):
    results = _mock_results([
        ("foo#0", 0.1, "A"),
        ("foo#1", 0.2, "B"),
        ("bar#-1", 0.3, "C"),
        ("foo#-1", 0.4, "D"),
    ])
    enriched = search._enrich_search_results(results, query="q")
    foo = next(r for r in enriched if r["item_key"] == "foo")
    bar = next(r for r in enriched if r["item_key"] == "bar")
    assert len(foo["passages"]) == 3
    assert len(bar["passages"]) == 1
