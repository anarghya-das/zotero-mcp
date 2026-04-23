"""Tests for the zotero_semantic_search MCP tool collection parameter."""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


class _FakeChromaClient:
    """Controls the has_documents_where probe used to detect empty scopes."""

    def __init__(self, has_docs_by_where=None):
        # Map a where-dict (as frozenset of items) to bool. Unknown => True,
        # so tests that don't care about empty-scope detection behave as before.
        self._has_docs = has_docs_by_where or {}

    def has_documents_where(self, where):
        key = frozenset(where.items())
        return self._has_docs.get(key, True)


class _FakeSearch:
    """Captures args to search.search() so the test can assert them."""

    last_call: dict = {}

    def __init__(self, chroma_client=None):
        self.chroma_client = chroma_client or _FakeChromaClient()

    def search(self, query, limit=10, filters=None):
        _FakeSearch.last_call = {"query": query, "limit": limit, "filters": filters}
        return {"results": [], "total_found": 0}


def _install_fake(monkeypatch, chroma_client=None):
    """Patch create_semantic_search on the module object present in sys.modules."""
    import zotero_mcp.semantic_search as ss
    fake = _FakeSearch(chroma_client=chroma_client)
    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: fake)
    return fake


def _tool():
    from zotero_mcp.tools.search import semantic_search
    return semantic_search


def test_collection_key_is_merged_into_filters(monkeypatch, dummy_ctx):
    _install_fake(monkeypatch)
    _tool()(query="neural models", collection="ABC12345", ctx=dummy_ctx)

    forwarded = _FakeSearch.last_call["filters"]
    assert forwarded == {"collection_key": "ABC12345"}


def test_collection_name_resolves_to_key_and_merges(monkeypatch, dummy_ctx):
    _install_fake(monkeypatch)
    import zotero_mcp.tools._helpers as zh
    monkeypatch.setattr(
        zh, "_resolve_collection_names",
        lambda zot, names, ctx=None: ["RESOLVED2"],
    )
    # Patch the Zotero client via sys.modules entry (see test_cli pattern).
    client_mod = sys.modules["zotero_mcp.client"]
    monkeypatch.setattr(client_mod, "get_zotero_client", lambda: object())

    _tool()(query="x", collection="Chapter 3", ctx=dummy_ctx)

    forwarded = _FakeSearch.last_call["filters"]
    assert forwarded == {"collection_key": "RESOLVED2"}


def test_collection_merges_alongside_existing_filters(monkeypatch, dummy_ctx):
    _install_fake(monkeypatch)
    _tool()(
        query="x",
        filters={"item_type": "journalArticle"},
        collection="ABC12345",
        ctx=dummy_ctx,
    )

    forwarded = _FakeSearch.last_call["filters"]
    assert forwarded["collection_key"] == "ABC12345"
    assert forwarded["item_type"] == "journalArticle"


def test_no_collection_leaves_filters_unchanged(monkeypatch, dummy_ctx):
    _install_fake(monkeypatch)
    _tool()(query="x", ctx=dummy_ctx)

    forwarded = _FakeSearch.last_call["filters"]
    # Previously passed as None; accept None or empty dict.
    assert forwarded is None or forwarded == {}


# ---------------------------------------------------------------------------
# Empty-scope detection: scope resolves to a real key, but nothing is indexed
# under it. Tool should short-circuit and return a structured prompt that
# gives the model two options to present to the user.
# ---------------------------------------------------------------------------


def test_empty_scope_returns_actionable_prompt(monkeypatch, dummy_ctx):
    chroma = _FakeChromaClient(has_docs_by_where={
        frozenset({"collection_key": "NEWCOLL1"}.items()): False,
    })
    _install_fake(monkeypatch, chroma_client=chroma)

    out = _tool()(query="sepsis", collection="NEWCOLL1", ctx=dummy_ctx)

    # The prompt must name the collection key so the user can act on it.
    assert "NEWCOLL1" in out
    # The fix-it command must appear verbatim so the model can suggest it.
    assert "update-db --collection NEWCOLL1" in out
    # The alternative tool must be named so the model can offer a fallback.
    assert "zotero_get_collection_items" in out
    # The message should clearly flag that no passages are indexed.
    assert "not indexed" in out.lower() or "no passages" in out.lower()
    # Tool should NOT have called search.search(): we short-circuit before.
    assert _FakeSearch.last_call == {} or _FakeSearch.last_call.get("query") != "sepsis"


def test_non_empty_scope_proceeds_normally(monkeypatch, dummy_ctx):
    chroma = _FakeChromaClient(has_docs_by_where={
        frozenset({"collection_key": "HASDATA1"}.items()): True,
    })
    _install_fake(monkeypatch, chroma_client=chroma)

    _tool()(query="sepsis", collection="HASDATA1", ctx=dummy_ctx)

    # Normal flow — search.search() was called with the merged filter.
    forwarded = _FakeSearch.last_call["filters"]
    assert forwarded == {"collection_key": "HASDATA1"}


def test_empty_scope_prompt_uses_resolved_name_when_available(monkeypatch, dummy_ctx):
    """If the user passed a collection name, echo it back in the prompt too."""
    chroma = _FakeChromaClient(has_docs_by_where={
        frozenset({"collection_key": "RESOLVED9"}.items()): False,
    })
    _install_fake(monkeypatch, chroma_client=chroma)

    import zotero_mcp.tools._helpers as zh
    monkeypatch.setattr(
        zh, "_resolve_collection_names",
        lambda zot, names, ctx=None: ["RESOLVED9"],
    )
    client_mod = sys.modules["zotero_mcp.client"]
    monkeypatch.setattr(client_mod, "get_zotero_client", lambda: object())

    out = _tool()(query="x", collection="Chapter 9", ctx=dummy_ctx)

    assert "RESOLVED9" in out
    assert "update-db --collection RESOLVED9" in out
