"""Tests for the zotero_semantic_search MCP tool collection parameter."""
from __future__ import annotations

import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )


class _FakeSearch:
    """Captures args to search.search() so the test can assert them."""

    last_call: dict = {}

    def search(self, query, limit=10, filters=None):
        _FakeSearch.last_call = {"query": query, "limit": limit, "filters": filters}
        return {"results": [], "total_found": 0}


def _install_fake(monkeypatch):
    """Patch create_semantic_search on the module object present in sys.modules."""
    import zotero_mcp.semantic_search as ss
    fake = _FakeSearch()
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
