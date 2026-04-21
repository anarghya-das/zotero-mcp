"""Test that the cross-encoder reranker is enabled by default.

Rationale: chunked retrieval benefits meaningfully from a cross-encoder
pass over the top candidates — the document-level cosine scores surface
many near-duplicates from the same paper, and the reranker's
query-passage pair scoring is much sharper for academic synthesis.
"""
from __future__ import annotations

import json
import sys

import pytest

if sys.version_info >= (3, 14):
    pytest.skip(
        "chromadb currently relies on pydantic v1 paths that are incompatible with Python 3.14+",
        allow_module_level=True,
    )

from zotero_mcp import semantic_search


class _FakeChroma:
    embedding_max_tokens = 8000

    class _Coll:
        metadata = {"chunked": True}
        def count(self):
            return 0
        def modify(self, *a, **kw):
            pass
    collection = _Coll()


def test_reranker_default_enabled_with_no_config(monkeypatch):
    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(chroma_client=_FakeChroma())
    assert s._reranker_config.get("enabled") is True


def test_config_file_can_still_opt_out(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "semantic_search": {
            "reranker": {"enabled": False},
        }
    }))

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(),
        config_path=str(cfg_path),
    )
    assert s._reranker_config.get("enabled") is False


def test_config_file_can_override_model(monkeypatch, tmp_path):
    cfg_path = tmp_path / "config.json"
    cfg_path.write_text(json.dumps({
        "semantic_search": {
            "reranker": {"enabled": True, "model": "some-other-model"},
        }
    }))

    monkeypatch.setattr(semantic_search, "get_zotero_client", lambda: object())
    s = semantic_search.ZoteroSemanticSearch(
        chroma_client=_FakeChroma(),
        config_path=str(cfg_path),
    )
    assert s._reranker_config.get("enabled") is True
    assert s._reranker_config.get("model") == "some-other-model"
