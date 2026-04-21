"""Tests for chunked output from _process_item_batch (Step 8).

After chunking lands, each paper produces 1 summary doc (chunk_idx=-1)
plus N fulltext chunks. Stale chunks from prior runs are purged before
upsert so re-indexing doesn't leave orphans.
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


# ---------------------------------------------------------------------------
# In-memory fake ChromaClient that exercises the delete_where + get_where path
# ---------------------------------------------------------------------------


class _FakeCollection:
    def __init__(self):
        self.data: dict[str, tuple[str, dict]] = {}  # id -> (doc, metadata)

    def get(self, where=None, ids=None, limit=None, include=None):
        if where:
            matched = [
                (k, v) for k, v in self.data.items()
                if all(v[1].get(wk) == wv for wk, wv in where.items())
            ]
            if limit:
                matched = matched[:limit]
            return {
                "ids": [k for k, _ in matched],
                "metadatas": [v[1] for _, v in matched],
                "documents": [v[0] for _, v in matched],
            }
        if ids is not None:
            hit = [(i, self.data[i]) for i in ids if i in self.data]
            return {
                "ids": [k for k, _ in hit],
                "metadatas": [v[1] for _, v in hit],
                "documents": [v[0] for _, v in hit],
            }
        return {
            "ids": list(self.data.keys()),
            "metadatas": [v[1] for v in self.data.values()],
            "documents": [v[0] for v in self.data.values()],
        }

    def delete(self, ids=None, where=None):
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
    def __init__(self, max_tokens=8000):
        self.collection = _FakeCollection()
        self.embedding_max_tokens = max_tokens

    def get_existing_ids(self, ids):
        return {i for i in ids if i in self.collection.data}

    def upsert_documents(self, documents, metadatas, ids):
        for doc, meta, doc_id in zip(documents, metadatas, ids):
            self.collection.data[doc_id] = (doc, meta)

    def truncate_text(self, text, max_tokens=None):
        return text

    def delete_documents_where(self, where):
        self.collection.delete(where=where)

    def get_document_metadata(self, doc_id):
        if doc_id in self.collection.data:
            return self.collection.data[doc_id][1]
        return None


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    return semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())


def _make_item(key, title="Paper", fulltext="", abstract="Short abstract"):
    return {
        "key": key,
        "data": {
            "title": title,
            "itemType": "journalArticle",
            "abstractNote": abstract,
            "creators": [],
            "fulltext": fulltext,
        },
    }


# ---------------------------------------------------------------------------
# Core chunking behavior
# ---------------------------------------------------------------------------


def test_no_fulltext_emits_only_summary_chunk(search):
    item = _make_item("KEYAAA01", title="Metadata-only paper")
    stats = search._process_item_batch([item])

    data = search.chroma_client.collection.data
    assert list(data.keys()) == ["KEYAAA01#-1"]
    meta = data["KEYAAA01#-1"][1]
    assert meta["chunk_idx"] == -1
    assert meta["chunk_total"] == 1
    assert stats["processed"] == 1


def test_long_fulltext_emits_summary_plus_chunks(search):
    """Fulltext triggers chunk_fulltext; summary (#-1) always present."""
    text = " ".join(str(i) for i in range(1, 4001))  # plenty of tokens
    item = _make_item("KEYBBB02", fulltext=text)

    search._process_item_batch([item])
    data = search.chroma_client.collection.data

    assert "KEYBBB02#-1" in data
    chunk_ids = [i for i in data if i.startswith("KEYBBB02#") and not i.endswith("#-1")]
    assert len(chunk_ids) >= 2, f"expected >= 2 fulltext chunks, got {chunk_ids}"

    # Chunk indices are 0, 1, 2, …
    idxs = sorted(data[i][1]["chunk_idx"] for i in chunk_ids)
    assert idxs == list(range(len(chunk_ids)))

    # chunk_total is consistent across all docs
    totals = {data[i][1]["chunk_total"] for i in data}
    assert len(totals) == 1
    assert next(iter(totals)) == len(data)


def test_stale_chunks_purged_before_upsert(search):
    """Pre-existing chunks for the same item_key must be dropped on re-index."""
    search.chroma_client.upsert_documents(
        documents=["stale"],
        metadatas=[{"item_key": "KEYCCC03", "chunk_idx": 99, "title": "old"}],
        ids=["KEYCCC03#99"],
    )

    item = _make_item("KEYCCC03", title="Fresh", fulltext="some new text")
    search._process_item_batch([item])

    data = search.chroma_client.collection.data
    assert "KEYCCC03#99" not in data
    assert "KEYCCC03#-1" in data


def test_legacy_bare_id_purged_on_rechunk(search):
    """A pre-chunking record with id=<key> (no `#` suffix) must also be removed."""
    search.chroma_client.upsert_documents(
        documents=["legacy bare id"],
        metadatas=[{"item_key": "KEYDDD04", "title": "legacy"}],
        ids=["KEYDDD04"],
    )
    item = _make_item("KEYDDD04", title="Fresh")
    search._process_item_batch([item])

    data = search.chroma_client.collection.data
    assert "KEYDDD04" not in data
    assert "KEYDDD04#-1" in data


def test_chunk_metadata_inherits_paper_fields(search):
    text = " ".join(str(i) for i in range(1, 4001))
    item = _make_item("KEYEEE05", title="Long Paper", fulltext=text)
    search._current_scope = ("COLLXXX1", "Chapter 1")

    search._process_item_batch([item])
    data = search.chroma_client.collection.data

    for doc_id, (_, meta) in data.items():
        assert meta["item_key"] == "KEYEEE05"
        assert meta["title"] == "Long Paper"
        assert meta["collection_key"] == "COLLXXX1"
        assert meta["collection_name"] == "Chapter 1"
        assert "chunk_idx" in meta
        assert "chunk_total" in meta


def test_force_rebuild_skips_purge(search, monkeypatch):
    """force_rebuild=True means the collection is already clean — no purge."""
    calls = []
    orig = search.chroma_client.delete_documents_where

    def _track(where):
        calls.append(where)
        return orig(where)

    monkeypatch.setattr(search.chroma_client, "delete_documents_where", _track)

    item = _make_item("KEYFFF06", fulltext="a b c")
    search._process_item_batch([item], force_rebuild=True)
    assert calls == []


def test_added_vs_updated_counting(search):
    """First run → added; second run on same key → updated."""
    item = _make_item("KEYGGG07", fulltext="some text")
    stats1 = search._process_item_batch([item])
    assert stats1["added"] == 1 and stats1["updated"] == 0

    stats2 = search._process_item_batch([item])
    assert stats2["added"] == 0 and stats2["updated"] == 1


def test_empty_key_skipped(search):
    stats = search._process_item_batch([_make_item("", fulltext="abc")])
    assert stats["skipped"] == 1
    assert stats["processed"] == 0
    assert search.chroma_client.collection.data == {}


def test_chunk_target_adapts_to_small_token_cap(monkeypatch):
    """Tiny embedding cap should still produce multiple chunks without crashing.

    Covers the default MiniLM (256 tokens) end of the spectrum. chunk_fulltext
    is told to target ~240 tokens so even short-cap models work.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    chroma = _FakeChroma(max_tokens=256)
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)

    text = " ".join(str(i) for i in range(1, 4001))
    item = _make_item("KEYHHH08", fulltext=text)
    search._process_item_batch([item])

    data = chroma.collection.data
    chunk_ids = [i for i in data if i.startswith("KEYHHH08#") and not i.endswith("#-1")]
    assert len(chunk_ids) >= 2


def test_summary_chunk_content_is_structured_text_only(search):
    """The #-1 chunk should contain title/abstract/creators, NOT fulltext body."""
    item = _make_item(
        "KEYIII09",
        title="Unique Title Xyzzy",
        abstract="Abstract Plugh Banana",
        fulltext="FULLTEXT_SENTINEL_TOKEN " * 50,
    )
    search._process_item_batch([item])

    summary = search.chroma_client.collection.data["KEYIII09#-1"][0]
    assert "Unique Title Xyzzy" in summary
    assert "Abstract Plugh Banana" in summary
    assert "FULLTEXT_SENTINEL_TOKEN" not in summary
