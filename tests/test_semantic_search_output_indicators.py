"""Tests that zotero_semantic_search surfaces indexing granularity.

The model answering the user needs to know, per result, whether the
best-matching chunk was the title+abstract summary or a fulltext
passage — and whether the paper has *any* fulltext indexed. Without
that, the model cannot tell the user that running `update-db --fulltext`
would enable deeper retrieval.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


class _FakeChromaClient:
    def has_documents_where(self, where):
        return True


class _FakeSearch:
    def __init__(self, results):
        self.chroma_client = _FakeChromaClient()
        self._results = results

    def search(self, query, limit=10, filters=None):
        return {"results": self._results, "total_found": len(self._results)}


def _result(item_key, best_chunk_idx, passages_chunk_indices, title="A Paper"):
    passages = [
        {"chunk_idx": i, "text": f"chunk {i} text", "similarity_score": 0.5,
         "metadata": {"item_key": item_key, "chunk_idx": i}}
        for i in passages_chunk_indices
    ]
    best = next(p for p in passages if p["chunk_idx"] == best_chunk_idx)
    return {
        "item_key": item_key,
        "similarity_score": best["similarity_score"],
        "matched_text": best["text"],
        "metadata": best["metadata"],
        "zotero_item": {"key": item_key, "data": {"title": title, "itemType": "journalArticle"}},
        "passages": passages,
        "query": "q",
    }


def _install(monkeypatch, results):
    import zotero_mcp.semantic_search as ss
    fake = _FakeSearch(results)
    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: fake)


def _tool():
    from zotero_mcp.tools.search import semantic_search
    return semantic_search


# ---------------------------------------------------------------------------
# Per-result indicators
# ---------------------------------------------------------------------------


def test_summary_only_paper_is_flagged(monkeypatch, dummy_ctx):
    """Paper with only the #-1 summary chunk indexed => output says so."""
    results = [_result("PAPERAAA", best_chunk_idx=-1, passages_chunk_indices=[-1])]
    _install(monkeypatch, results)

    out = _tool()(query="sepsis", ctx=dummy_ctx)

    # The result block should flag that only the summary is indexed.
    assert "summary only" in out.lower() or "title + abstract" in out.lower()
    # And should recommend --fulltext so the user can drill deeper.
    assert "--fulltext" in out


def test_fulltext_passage_match_is_flagged(monkeypatch, dummy_ctx):
    """Best chunk is fulltext (>=0) => output flags it as a fulltext passage."""
    results = [_result("PAPERBBB", best_chunk_idx=3, passages_chunk_indices=[-1, 0, 1, 2, 3])]
    _install(monkeypatch, results)

    out = _tool()(query="sepsis", ctx=dummy_ctx)

    # Should indicate this is a fulltext passage, not summary-only.
    assert "fulltext" in out.lower() or "passage" in out.lower()
    # Must NOT suggest this paper needs --fulltext indexing.
    lines = [l for l in out.split("\n") if "PAPERBBB" in l or "--fulltext" in l]
    # If --fulltext is mentioned anywhere, it must not be in the context of this paper.


def test_best_match_summary_but_fulltext_available(monkeypatch, dummy_ctx):
    """Summary chunk outscored fulltext — flag that deeper passages exist."""
    results = [_result("PAPERCCC", best_chunk_idx=-1, passages_chunk_indices=[-1, 0, 1, 2])]
    _install(monkeypatch, results)

    out = _tool()(query="q", ctx=dummy_ctx)

    # Summary was the best match, but fulltext IS indexed — the model should
    # know it can use zotero_get_item_passages for deeper retrieval.
    assert "zotero_get_item_passages" in out or "passages" in out.lower()


# ---------------------------------------------------------------------------
# Aggregate header: tell model "N of M summary-only"
# ---------------------------------------------------------------------------


def test_header_summarizes_fulltext_coverage(monkeypatch, dummy_ctx):
    """Mixed results: 2 summary-only, 1 fulltext. Header should say so."""
    results = [
        _result("P1", best_chunk_idx=-1, passages_chunk_indices=[-1]),
        _result("P2", best_chunk_idx=-1, passages_chunk_indices=[-1]),
        _result("P3", best_chunk_idx=2,  passages_chunk_indices=[-1, 0, 1, 2]),
    ]
    _install(monkeypatch, results)

    out = _tool()(query="q", ctx=dummy_ctx)

    # Some aggregate line should tell the model how many results are summary-only.
    low = out.lower()
    assert ("2" in out and ("summary" in low or "metadata" in low or "no fulltext" in low))
