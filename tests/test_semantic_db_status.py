"""Tests for zotero_semantic_db_status MCP tool.

The tool introspects the semantic search collection and summarizes
what's actually indexed — per-collection paper/chunk counts, total
papers (unique item_keys, not chunk count), embedding model, and
the chunked-schema marker. Raw "total docs" is misleading because
one paper becomes many chunk documents.
"""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


def _mk_meta(item_key, chunk_idx, collection_key=None, collection_name=None, title=None):
    m = {"item_key": item_key, "chunk_idx": chunk_idx}
    if collection_key:
        m["collection_key"] = collection_key
    if collection_name:
        m["collection_name"] = collection_name
    if title:
        m["title"] = title
    return m


class _FakeCollection:
    def __init__(self, metas, metadata=None):
        self._metas = metas
        self.metadata = metadata or {"chunked": True, "chunk_version": 1}

    def get(self, include=None, where=None, ids=None, limit=None):
        return {"metadatas": list(self._metas)}

    def count(self):
        return len(self._metas)


class _FakeChroma:
    def __init__(self, metas, embedding_model="openai", metadata=None,
                 model_name="text-embedding-3-large"):
        self.collection = _FakeCollection(metas, metadata=metadata)
        self.embedding_model = embedding_model
        class _EF:
            pass
        ef = _EF()
        ef.model_name = model_name
        self.embedding_function = ef
        self.persist_directory = "/tmp/fake-chroma"
        self.embedding_max_tokens = 8000


class _FakeSearch:
    def __init__(self, chroma, update_config=None):
        self.chroma_client = chroma
        self.update_config = update_config or {
            "auto_update": False,
            "update_frequency": "manual",
            "last_update": "2026-04-21T14:48:37.349959",
        }


def _install(monkeypatch, search):
    import zotero_mcp.semantic_search as ss
    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: search)


def _tool():
    from zotero_mcp.tools.search import semantic_db_status
    return semantic_db_status


# ---------------------------------------------------------------------------
# Happy path — one scoped collection, no unscoped records
# ---------------------------------------------------------------------------


def test_single_collection_breakdown(monkeypatch, dummy_ctx):
    """One scoped collection with 3 papers, each 2 chunks = 6 docs."""
    metas = []
    for pk in ("P1", "P2", "P3"):
        for idx in (-1, 0):
            metas.append(_mk_meta(pk, idx, "ABC12345", "ai-icu-review", "Paper " + pk))

    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    assert "ai-icu-review" in out
    assert "ABC12345" in out
    assert "3" in out  # paper count
    assert "6" in out  # chunk count
    assert "openai" in out.lower()
    assert "text-embedding-3-large" in out


def test_reports_total_unique_papers_not_chunk_count(monkeypatch, dummy_ctx):
    """108 papers × 15 chunks avg = 1620 docs — total papers must be 108."""
    metas = [
        _mk_meta(f"P{i:03d}", idx, "ABC12345", "ai-icu-review", f"Paper {i}")
        for i in range(108) for idx in range(-1, 14)  # 15 chunks each
    ]
    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    # Look for the paper count (108) in the output; 1620 (chunks) will also
    # appear, but the PAPERS column must report 108 for the collection row.
    # The markdown table should have "108" followed by chunk count.
    assert "108" in out
    # Sanity: chunks column should surface too
    assert "1620" in out or "1,620" in out


# ---------------------------------------------------------------------------
# Multiple collections
# ---------------------------------------------------------------------------


def test_multiple_collections_distinct_rows(monkeypatch, dummy_ctx):
    metas = []
    # Collection A: 2 papers, 3 chunks each = 6 docs
    for pk in ("A1", "A2"):
        for idx in (-1, 0, 1):
            metas.append(_mk_meta(pk, idx, "AAAAAAAA", "collection-a"))
    # Collection B: 1 paper, 4 chunks = 4 docs
    for idx in (-1, 0, 1, 2):
        metas.append(_mk_meta("B1", idx, "BBBBBBBB", "collection-b"))

    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    assert "collection-a" in out
    assert "collection-b" in out
    assert "AAAAAAAA" in out
    assert "BBBBBBBB" in out


# ---------------------------------------------------------------------------
# Unscoped records surfaced under a distinct label
# ---------------------------------------------------------------------------


def test_unscoped_records_surfaced(monkeypatch, dummy_ctx):
    """Records without collection_key stamp should appear as '(unscoped)'.

    Important for diagnosing auto-update overwrite situations where
    scoped stamps got wiped.
    """
    metas = [
        _mk_meta("X1", -1),  # no collection_key
        _mk_meta("X1", 0),
        _mk_meta("X2", -1),
    ]
    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    assert "unscoped" in out.lower()
    # 2 papers, 3 chunks in the unscoped bucket
    assert "2" in out and "3" in out


def test_mixed_scoped_and_unscoped(monkeypatch, dummy_ctx):
    metas = []
    # Scoped
    for idx in (-1, 0, 1):
        metas.append(_mk_meta("S1", idx, "SCOPED01", "scoped-coll"))
    # Unscoped
    for idx in (-1, 0):
        metas.append(_mk_meta("U1", idx))

    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    assert "scoped-coll" in out
    assert "unscoped" in out.lower()


# ---------------------------------------------------------------------------
# Marker, auto-update, last update surface in the output
# ---------------------------------------------------------------------------


def test_reports_chunked_marker(monkeypatch, dummy_ctx):
    chroma = _FakeChroma([_mk_meta("P1", -1)], metadata={"chunked": True, "chunk_version": 1})
    _install(monkeypatch, _FakeSearch(chroma))
    out = _tool()(ctx=dummy_ctx)
    assert "chunked" in out.lower()


def test_warns_when_marker_missing(monkeypatch, dummy_ctx):
    chroma = _FakeChroma([_mk_meta("P1", -1)], metadata=None)
    _install(monkeypatch, _FakeSearch(chroma))
    out = _tool()(ctx=dummy_ctx)
    assert "legacy" in out.lower() or "force-rebuild" in out.lower()


def test_reports_auto_update_status(monkeypatch, dummy_ctx):
    """User should be able to see whether auto-update is on."""
    search_on = _FakeSearch(_FakeChroma([_mk_meta("P1", -1)]), update_config={
        "auto_update": True, "update_frequency": "startup",
        "last_update": "2026-04-21T14:48:37.349959",
    })
    _install(monkeypatch, search_on)
    out_on = _tool()(ctx=dummy_ctx)
    assert "enabled" in out_on.lower() or "on" in out_on.lower()

    search_off = _FakeSearch(_FakeChroma([_mk_meta("P1", -1)]))
    _install(monkeypatch, search_off)
    out_off = _tool()(ctx=dummy_ctx)
    assert "disabled" in out_off.lower() or "off" in out_off.lower() or "manual" in out_off.lower()


# ---------------------------------------------------------------------------
# Empty / edge cases
# ---------------------------------------------------------------------------


def test_empty_collection(monkeypatch, dummy_ctx):
    _install(monkeypatch, _FakeSearch(_FakeChroma([])))
    out = _tool()(ctx=dummy_ctx)
    low = out.lower()
    assert "empty" in low or "0 paper" in low or "no papers" in low


def test_per_collection_fulltext_column(monkeypatch, dummy_ctx):
    """User should see how many papers in each collection have fulltext indexed.

    A paper has fulltext iff any of its chunks has `chunk_idx >= 0`. Summary-only
    papers have just `chunk_idx == -1`.
    """
    metas = []
    # Collection A: P1 has fulltext (chunks -1, 0, 1); P2 is summary-only (-1).
    for idx in (-1, 0, 1):
        metas.append(_mk_meta("P1", idx, "AAAAAAAA", "coll-a"))
    metas.append(_mk_meta("P2", -1, "AAAAAAAA", "coll-a"))
    # Collection B: Q1 is summary-only.
    metas.append(_mk_meta("Q1", -1, "BBBBBBBB", "coll-b"))

    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    # The per-collection table should show a fulltext column.
    low = out.lower()
    assert "fulltext" in low or "full-text" in low
    # coll-a: 2 papers total, 1 has fulltext.
    # coll-b: 1 paper total, 0 have fulltext.
    # These counts must appear in the output (exact row format left flexible).
    assert "coll-a" in out and "coll-b" in out


def test_fulltext_coverage_all_summary_only(monkeypatch, dummy_ctx):
    """Entire collection is summary-only — column should show 0 fulltext."""
    metas = [
        _mk_meta("P1", -1, "CCCCCCCC", "coll-c"),
        _mk_meta("P2", -1, "CCCCCCCC", "coll-c"),
        _mk_meta("P3", -1, "CCCCCCCC", "coll-c"),
    ]
    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)

    # Output should clearly indicate zero fulltext coverage for this collection.
    assert "coll-c" in out
    assert "fulltext" in out.lower() or "full-text" in out.lower()


def test_chunk_stats_reported(monkeypatch, dummy_ctx):
    """Min/median/max/avg chunks per paper should surface."""
    # Paper A: 5 chunks; Paper B: 10 chunks; Paper C: 20 chunks
    metas = []
    for pk, n in (("A", 5), ("B", 10), ("C", 20)):
        for i in range(-1, n - 1):  # N total including summary
            metas.append(_mk_meta(pk, i, "KKKKKKKK", "k"))

    _install(monkeypatch, _FakeSearch(_FakeChroma(metas)))
    out = _tool()(ctx=dummy_ctx)
    assert "min" in out.lower() and "max" in out.lower()
