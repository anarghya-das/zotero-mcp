"""Tests for `zotero-mcp update-db` CLI collection flags.

Exercises argparse wiring end-to-end: invokes cli.main() with sys.argv
patched, mocks out create_semantic_search and any pyzotero calls, and
asserts the correct collection_key / collection_name reach
search.update_database(...).
"""
from __future__ import annotations

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Environment in which cli.main() can run without touching the world."""
    # Silence setup_zotero_environment side effects.
    from zotero_mcp import cli

    monkeypatch.setattr(cli, "setup_zotero_environment", lambda: None)

    # Stub out config file logic by pointing HOME at a temp dir.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)

    # Prevent get_zotero_client's env-var check from firing when another test
    # has cleared ZOTERO_API_KEY / LIBRARY_ID. The individual tests still
    # replace the client with a mock via monkeypatch.setattr below.
    monkeypatch.setenv("ZOTERO_LOCAL", "true")

    return cli


def _run_cli(argv, monkeypatch):
    """Invoke cli.main() with argv, catching SystemExit."""
    monkeypatch.setattr(sys, "argv", ["zotero-mcp"] + argv)
    from zotero_mcp import cli

    try:
        cli.main()
    except SystemExit:
        pass


def _patch_client(monkeypatch, replacement):
    """Patch get_zotero_client via sys.modules.

    Must NOT import `zotero_mcp.client as zm_client` and then setattr on
    that — another test file (test_relations_field.py:26) replaces
    sys.modules["zotero_mcp.client"] at collection time, so the module
    object bound by `import ... as` may diverge from the entry that
    `from zotero_mcp.client import ...` resolves against.
    """
    import sys
    client_mod = sys.modules["zotero_mcp.client"]
    monkeypatch.setattr(client_mod, "get_zotero_client", replacement)


def test_collection_key_flag_reaches_update_database(cli_env, monkeypatch):
    """--collection ABC12345 should be forwarded to search.update_database()."""
    fake_search = MagicMock()
    fake_search.update_database.return_value = {"total_items": 0, "processed_items": 0}

    import zotero_mcp.semantic_search as ss
    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: fake_search)

    class _Zot:
        def collection(self, key):
            return {"data": {"name": "Chapter 1"}}

    _patch_client(monkeypatch, lambda: _Zot())

    _run_cli(["update-db", "--collection", "ABC12345"], monkeypatch)

    fake_search.update_database.assert_called_once()
    kwargs = fake_search.update_database.call_args.kwargs
    assert kwargs["collection_key"] == "ABC12345"
    assert kwargs["collection_name"] == "Chapter 1"


def test_collection_name_resolves_to_key(cli_env, monkeypatch):
    """--collection-name 'Chapter 3' should resolve via _resolve_collection_names."""
    fake_search = MagicMock()
    fake_search.update_database.return_value = {"total_items": 0, "processed_items": 0}

    import zotero_mcp.semantic_search as ss
    import zotero_mcp.tools._helpers as zh

    monkeypatch.setattr(ss, "create_semantic_search", lambda *a, **kw: fake_search)
    _patch_client(monkeypatch, lambda: object())
    monkeypatch.setattr(
        zh, "_resolve_collection_names", lambda zot, names, ctx=None: ["RESOLVED1"]
    )

    _run_cli(["update-db", "--collection-name", "Chapter 3"], monkeypatch)

    fake_search.update_database.assert_called_once()
    kwargs = fake_search.update_database.call_args.kwargs
    assert kwargs["collection_key"] == "RESOLVED1"
    assert kwargs["collection_name"] == "Chapter 3"


def test_no_collection_flag_leaves_scope_unset(cli_env, monkeypatch):
    fake_search = MagicMock()
    fake_search.update_database.return_value = {"total_items": 0, "processed_items": 0}
    monkeypatch.setattr(
        "zotero_mcp.semantic_search.create_semantic_search",
        lambda *a, **kw: fake_search,
    )

    _run_cli(["update-db"], monkeypatch)

    fake_search.update_database.assert_called_once()
    kwargs = fake_search.update_database.call_args.kwargs
    assert kwargs.get("collection_key") is None
    assert kwargs.get("collection_name") is None


def test_both_flags_error_is_friendly(cli_env, monkeypatch, capsys):
    """Passing both --collection and --collection-name is ambiguous; CLI should reject."""
    monkeypatch.setattr(
        "zotero_mcp.semantic_search.create_semantic_search",
        lambda *a, **kw: MagicMock(),
    )

    _run_cli(
        ["update-db", "--collection", "ABC12345", "--collection-name", "X"],
        monkeypatch,
    )
    err = capsys.readouterr().err
    assert "--collection" in err and "--collection-name" in err
