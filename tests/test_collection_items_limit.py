"""Tests for zotero_get_collection_items display limit behavior.

Background: the tool hard-capped `limit` at 100 via `_normalize_limit`
regardless of `detail` level. That broke workflows where the model
needed to list all items in a mid-to-large collection (e.g. 173 papers
for literature-review triage). `keys_only` output is ~40 bytes per item
so 100 is an artificially tight ceiling; `full` output with abstracts
reasonably stays at 100.

Policy: the ceiling now scales with detail level:
- keys_only → high ceiling (thousands OK — output is tiny)
- summary   → moderate ceiling (hundreds)
- full      → keep 100 (each item ~500 tokens)
"""
from __future__ import annotations

import sys

import pytest

# Reuse fixtures and helpers from the token-optimization suite.
from test_token_optimization import (  # type: ignore
    CollectionFakeZotero,
    DummyContext,
    _make_attachment,
    _make_parent,
)


class PaginatedFakeZotero(CollectionFakeZotero):
    """CollectionFakeZotero honours `start`/`limit` so `_paginate` terminates.

    The parent fake ignores pagination kwargs and returns the full list on
    every call — fine for fixtures with < 100 items, but against larger
    collections `_helpers._paginate` loops forever because `len(batch)`
    stays >= page_size every iteration.
    """

    def collection_items(self, key, start=0, limit=None, **kwargs):
        matching = [
            it for it in self._all_items
            if key in it.get("data", {}).get("collections", [])
        ]
        if limit is None:
            return matching[start:]
        return matching[start:start + limit]


def _seed(n: int) -> PaginatedFakeZotero:
    zot = PaginatedFakeZotero()
    items = []
    for i in range(n):
        items.append(_make_parent(f"P{i:04d}", f"Paper {i}", abstract=f"Abstract {i}"))
        # Half have PDFs just to keep the attachment-summary code path exercised.
        if i % 2 == 0:
            items.append(_make_attachment(f"A{i}", f"P{i:04d}", "application/pdf", f"paper{i}.pdf"))
    zot._all_items = items
    return zot


@pytest.fixture
def dummy_ctx():
    return DummyContext()


def _tool():
    from zotero_mcp.tools.retrieval import get_collection_items
    return get_collection_items


def _patch_client(monkeypatch, zot):
    monkeypatch.setattr(
        "zotero_mcp.tools.retrieval._client.get_zotero_client", lambda: zot
    )


# ---------------------------------------------------------------------------
# keys_only: can return well past 100 items
# ---------------------------------------------------------------------------


def test_keys_only_returns_all_173_items_when_requested(monkeypatch, dummy_ctx):
    zot = _seed(173)
    _patch_client(monkeypatch, zot)

    out = _tool()(collection_key="COL1", detail="keys_only", limit=200, ctx=dummy_ctx)

    # The last item key must appear — i.e., the ceiling did not clip us at 100.
    assert "`P0172`" in out, "Last item missing — limit was clipped below 173"
    # And the total count line must report 173.
    assert "173" in out


def test_keys_only_limit_500_not_clamped_to_100(monkeypatch, dummy_ctx):
    zot = _seed(250)
    _patch_client(monkeypatch, zot)

    out = _tool()(collection_key="COL1", detail="keys_only", limit=500, ctx=dummy_ctx)

    # All 250 should render since 500 > 250 and the ceiling accommodates 500.
    assert "`P0249`" in out
    # Sanity: "250 items" must appear in the header line.
    assert "250" in out


def test_full_ceiling_permits_200_items(monkeypatch, dummy_ctx):
    """The `full` ceiling is 300 (sized for 1M-context consumers), so a
    200-paper review collection must render completely at detail='full'."""
    zot = _seed(200)
    _patch_client(monkeypatch, zot)

    out = _tool()(collection_key="COL1", detail="full", limit=200, ctx=dummy_ctx)

    # Last item must appear — previous 100-ceiling would have clipped it.
    assert "Paper 199" in out
    # Header reports the full 200.
    assert "200 items" in out


# ---------------------------------------------------------------------------
# summary: ceiling is higher than 100 but lower than keys_only
# ---------------------------------------------------------------------------


def test_summary_allows_more_than_100(monkeypatch, dummy_ctx):
    zot = _seed(200)
    _patch_client(monkeypatch, zot)

    out = _tool()(collection_key="COL1", detail="summary", limit=200, ctx=dummy_ctx)

    # Item 100 (zero-indexed 100) must appear — was previously clipped.
    assert "`P0100`" in out or "Paper 100" in out


# ---------------------------------------------------------------------------
# full: retains the tight 100 ceiling (each item ~500 tokens with abstract)
# ---------------------------------------------------------------------------


def test_full_still_caps_at_ceiling(monkeypatch, dummy_ctx):
    """`full` ceiling is 300 — bigger than that must clip to protect context."""
    zot = _seed(400)
    _patch_client(monkeypatch, zot)

    out = _tool()(collection_key="COL1", detail="full", limit=500, ctx=dummy_ctx)

    # At 400 items with ceiling 300, item 399 must NOT appear but 299 should.
    assert "Paper 399" not in out
    assert "Paper 299" in out


# ---------------------------------------------------------------------------
# Offset / pagination: model can walk past the ceiling by re-calling with
# `offset`. Essential for libraries that exceed the per-detail ceiling.
# ---------------------------------------------------------------------------


def test_offset_skips_first_n_items(monkeypatch, dummy_ctx):
    """offset=100 returns items starting from index 100."""
    zot = _seed(300)
    _patch_client(monkeypatch, zot)

    out = _tool()(
        collection_key="COL1", detail="keys_only",
        limit=50, offset=100, ctx=dummy_ctx,
    )

    # Items 100..149 should appear; 0..99 and 150+ should not.
    assert "`P0100`" in out
    assert "`P0149`" in out
    assert "`P0099`" not in out
    assert "`P0150`" not in out


def test_offset_plus_limit_defines_window(monkeypatch, dummy_ctx):
    """A page-2 request (offset=300, limit=300) returns items 300..599."""
    zot = _seed(600)
    _patch_client(monkeypatch, zot)

    out = _tool()(
        collection_key="COL1", detail="full",
        limit=300, offset=300, ctx=dummy_ctx,
    )

    assert "Paper 300" in out
    assert "Paper 599" in out
    assert "Paper 299" not in out


def test_offset_default_zero_matches_unparametrized(monkeypatch, dummy_ctx):
    """offset=0 (default) behaves exactly like not passing offset."""
    zot1 = _seed(50)
    zot2 = _seed(50)
    _patch_client(monkeypatch, zot1)
    out_default = _tool()(collection_key="COL1", detail="keys_only", ctx=dummy_ctx)
    _patch_client(monkeypatch, zot2)
    out_explicit_zero = _tool()(
        collection_key="COL1", detail="keys_only", offset=0, ctx=dummy_ctx,
    )
    assert out_default == out_explicit_zero


def test_offset_past_end_returns_no_items_on_this_page(monkeypatch, dummy_ctx):
    """offset > total parent count should produce an empty page with a clear hint."""
    zot = _seed(50)
    _patch_client(monkeypatch, zot)

    out = _tool()(
        collection_key="COL1", detail="keys_only",
        limit=50, offset=500, ctx=dummy_ctx,
    )

    # The header should still show the true total (50) so the model knows
    # it overshot.
    assert "50" in out
    # And no paper lines for P0000..P0049 should render.
    for i in range(50):
        assert f"`P{i:04d}`" not in out


def test_footer_shows_range_and_next_offset_hint(monkeypatch, dummy_ctx):
    """When results are paginated, the footer must report the range shown
    and the next offset so the model can walk the list."""
    zot = _seed(500)
    _patch_client(monkeypatch, zot)

    out = _tool()(
        collection_key="COL1", detail="keys_only",
        limit=100, offset=100, ctx=dummy_ctx,
    )

    # Footer must surface the range in a form the model can act on.
    # We accept either "items 100..199" or "offset=200" style hints.
    low = out.lower()
    assert "100" in out and ("199" in out or "next" in low or "offset" in low)
