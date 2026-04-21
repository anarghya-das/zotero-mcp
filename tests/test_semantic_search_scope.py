"""Tests for collection-scoped wiring inside ZoteroSemanticSearch.

Covers:
- _create_metadata stamps collection_key / collection_name when scope is set.
- _get_items_from_local_db passes collection_key through to LocalZoteroReader.
- _get_items_from_api switches to zot.collection_items when scope is set.
- update_database stashes scope on self._current_scope.
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


class _FakeChroma:
    """Minimal ChromaClient stub sufficient to instantiate ZoteroSemanticSearch."""

    def __init__(self):
        self.embedding_max_tokens = 8000

    def get_document_metadata(self, doc_id):
        return None

    def get_existing_ids(self, ids):
        return set()

    def upsert_documents(self, documents, metadatas, ids):
        pass

    def truncate_text(self, text, max_tokens=None):
        return text


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    return semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())


# ---------------------------------------------------------------------------
# _create_metadata scope stamping
# ---------------------------------------------------------------------------


def _sample_item():
    return {
        "key": "ITEMAA01",
        "data": {
            "title": "A Paper",
            "itemType": "journalArticle",
            "abstractNote": "",
            "creators": [],
            "dateAdded": "",
            "dateModified": "",
        },
    }


def test_create_metadata_no_scope_has_no_collection_fields(search):
    meta = search._create_metadata(_sample_item())
    assert "collection_key" not in meta
    assert "collection_name" not in meta


def test_create_metadata_stamps_scope_when_set(search):
    search._current_scope = ("COLLXXX1", "Chapter 1")
    meta = search._create_metadata(_sample_item())
    assert meta["collection_key"] == "COLLXXX1"
    assert meta["collection_name"] == "Chapter 1"


def test_create_metadata_stamps_key_only_when_name_missing(search):
    search._current_scope = ("COLLXXX1", None)
    meta = search._create_metadata(_sample_item())
    assert meta["collection_key"] == "COLLXXX1"
    # Name is empty-string or absent — either is acceptable; assert not populated.
    assert not meta.get("collection_name")


# ---------------------------------------------------------------------------
# _get_items_from_local_db passes collection_key to the reader
# ---------------------------------------------------------------------------


class _FakeReader:
    """Stands in for LocalZoteroReader to capture the kwargs passed in."""

    last_kwargs: dict = {}

    def __init__(self, db_path=None, pdf_max_pages=None, pdf_timeout=30):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get_items_with_text(self, limit=None, include_fulltext=False, collection_key=None, **kw):
        _FakeReader.last_kwargs = {
            "limit": limit,
            "include_fulltext": include_fulltext,
            "collection_key": collection_key,
        }
        return []

    def get_fulltext_meta_for_item(self, item_id):
        return []

    def extract_fulltext_for_item(self, item_id):
        return None


def test_local_db_passes_collection_key_to_reader(monkeypatch, search):
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", _FakeReader)
    # Bypass local-mode check by calling the private method directly.
    search._get_items_from_local_db(limit=None, collection_key="COLLXXX1")
    assert _FakeReader.last_kwargs["collection_key"] == "COLLXXX1"


def test_local_db_without_collection_key_passes_none(monkeypatch, search):
    monkeypatch.setattr(semantic_search, "LocalZoteroReader", _FakeReader)
    search._get_items_from_local_db(limit=None)
    assert _FakeReader.last_kwargs["collection_key"] is None


# ---------------------------------------------------------------------------
# _get_items_from_api switches to collection_items when scope is set
# ---------------------------------------------------------------------------


class _RecordingZot:
    def __init__(self):
        self.items_calls = []
        self.collection_items_calls = []

    def items(self, **kwargs):
        self.items_calls.append(kwargs)
        return []  # empty stops the paging loop

    def collection_items(self, key, **kwargs):
        self.collection_items_calls.append((key, kwargs))
        return []


def test_api_uses_collection_items_when_scoped(search):
    zot = _RecordingZot()
    search.zotero_client = zot
    search._get_items_from_api(limit=None, collection_key="COLLXXX1")
    assert zot.collection_items_calls, "expected collection_items to be called"
    assert zot.collection_items_calls[0][0] == "COLLXXX1"
    assert not zot.items_calls, "items() must not be called in scoped mode"


def test_api_uses_items_when_unscoped(search):
    zot = _RecordingZot()
    search.zotero_client = zot
    search._get_items_from_api(limit=None)
    assert zot.items_calls, "expected items() to be called"
    assert not zot.collection_items_calls


# ---------------------------------------------------------------------------
# update_database stashes scope
# ---------------------------------------------------------------------------


def test_update_database_sets_current_scope(monkeypatch, search):
    """update_database should stash scope before extraction runs.

    We observe via the _get_items_from_source hook.
    """
    captured = {}

    def _fake_source(self, *, limit, extract_fulltext, chroma_client, force_rebuild, collection_key=None):
        captured["collection_key"] = collection_key
        captured["current_scope"] = getattr(self, "_current_scope", None)
        return []

    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch,
        "_get_items_from_source",
        _fake_source,
    )

    search.update_database(collection_key="COLLXXX1", collection_name="Chapter 1")

    assert captured["collection_key"] == "COLLXXX1"
    assert captured["current_scope"] == ("COLLXXX1", "Chapter 1")


# ---------------------------------------------------------------------------
# Step 6 — scoped force-rebuild purges by collection_key only
# ---------------------------------------------------------------------------


class _RecordingChroma(_FakeChroma):
    """Captures which teardown path update_database took."""

    def __init__(self):
        super().__init__()
        self.reset_calls = 0
        self.where_deletes: list[dict] = []

    def reset_collection(self):
        self.reset_calls += 1

    def delete_documents_where(self, where):
        self.where_deletes.append(where)


def _stub_source(monkeypatch):
    """Short-circuit _get_items_from_source so update_database does no work."""
    monkeypatch.setattr(
        semantic_search.ZoteroSemanticSearch,
        "_get_items_from_source",
        lambda self, **kw: [],
    )


def test_scoped_force_rebuild_deletes_by_collection_only(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    chroma = _RecordingChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    _stub_source(monkeypatch)

    search.update_database(
        force_full_rebuild=True,
        collection_key="COLLXXX1",
        collection_name="Chapter 1",
    )

    assert chroma.reset_calls == 0, "scoped rebuild must not reset the whole collection"
    assert chroma.where_deletes == [{"collection_key": "COLLXXX1"}]


def test_unscoped_force_rebuild_still_resets_whole_collection(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    chroma = _RecordingChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    _stub_source(monkeypatch)

    search.update_database(force_full_rebuild=True)

    assert chroma.reset_calls == 1
    assert chroma.where_deletes == []


def test_scoped_no_rebuild_does_not_delete(monkeypatch):
    """update_database --collection without --force-rebuild should NOT purge.

    Incremental updates rely on item-level upserts, not collection-wide delete.
    """
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    chroma = _RecordingChroma()
    search = semantic_search.ZoteroSemanticSearch(chroma_client=chroma)
    _stub_source(monkeypatch)

    search.update_database(force_full_rebuild=False, collection_key="COLLXXX1")

    assert chroma.reset_calls == 0
    assert chroma.where_deletes == []
