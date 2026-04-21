"""Tests for LocalZoteroReader collection-scoped filtering.

Builds a minimal in-memory Zotero SQLite schema with items across multiple
collections and verifies that `get_items_with_text(collection_key=...)`
returns only matching items without row duplication.
"""
from __future__ import annotations

import sqlite3

import pytest

from zotero_mcp.local_db import LocalZoteroReader


# ---------------------------------------------------------------------------
# Schema + seed helpers
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE items (
    itemID INTEGER PRIMARY KEY,
    key TEXT,
    itemTypeID INTEGER,
    dateAdded TEXT,
    dateModified TEXT
);
CREATE TABLE itemData (
    itemID INTEGER,
    fieldID INTEGER,
    valueID INTEGER
);
CREATE TABLE itemDataValues (
    valueID INTEGER PRIMARY KEY,
    value TEXT
);
CREATE TABLE creators (
    creatorID INTEGER PRIMARY KEY,
    firstName TEXT,
    lastName TEXT
);
CREATE TABLE itemCreators (
    itemID INTEGER,
    creatorID INTEGER,
    creatorTypeID INTEGER,
    orderIndex INTEGER
);
CREATE TABLE itemNotes (
    itemID INTEGER,
    parentItemID INTEGER,
    note TEXT
);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
CREATE TABLE collections (
    collectionID INTEGER PRIMARY KEY,
    key TEXT,
    collectionName TEXT,
    parentCollectionID INTEGER
);
CREATE TABLE collectionItems (
    collectionID INTEGER,
    itemID INTEGER
);
"""


def _seed(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(_SCHEMA)
    cur.executemany(
        "INSERT INTO itemTypes(itemTypeID, typeName) VALUES (?, ?)",
        [(1, "journalArticle"), (2, "attachment"), (3, "note")],
    )
    # fieldID=1 title, fieldID=2 abstract, fieldID=16 extra
    cur.executemany(
        "INSERT INTO fields(fieldID, fieldName) VALUES (?, ?)",
        [(1, "title"), (2, "abstractNote"), (16, "extra"), (99, "DOI")],
    )

    # 4 items: AA in coll X, BB in coll X, CC in coll Y, DD in BOTH colls
    items = [
        (10, "ITEMAA01"),
        (20, "ITEMBB02"),
        (30, "ITEMCC03"),
        (40, "ITEMDD04"),
    ]
    cur.executemany(
        "INSERT INTO items(itemID, key, itemTypeID, dateAdded, dateModified) "
        "VALUES (?, ?, 1, '2026-01-01', '2026-01-01')",
        items,
    )

    # Titles
    cur.executemany(
        "INSERT INTO itemDataValues(valueID, value) VALUES (?, ?)",
        [
            (100, "Paper AA"),
            (200, "Paper BB"),
            (300, "Paper CC"),
            (400, "Paper DD"),
        ],
    )
    cur.executemany(
        "INSERT INTO itemData(itemID, fieldID, valueID) VALUES (?, 1, ?)",
        [(10, 100), (20, 200), (30, 300), (40, 400)],
    )

    # 2 collections
    cur.executemany(
        "INSERT INTO collections(collectionID, key, collectionName, parentCollectionID) "
        "VALUES (?, ?, ?, NULL)",
        [(1, "COLLXXX1", "Chapter 1"), (2, "COLLYYY2", "Chapter 2")],
    )
    # Membership: AA→X, BB→X, CC→Y, DD→X AND Y
    cur.executemany(
        "INSERT INTO collectionItems(collectionID, itemID) VALUES (?, ?)",
        [(1, 10), (1, 20), (2, 30), (1, 40), (2, 40)],
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Reader factory — bypass _find_zotero_db by pointing at our temp DB.
# ---------------------------------------------------------------------------


@pytest.fixture
def reader_factory(tmp_path, monkeypatch):
    """Return a callable that gives a LocalZoteroReader bound to a seeded SQLite."""

    db_path = tmp_path / "zotero.sqlite"

    def _make():
        conn = sqlite3.connect(str(db_path))
        _seed(conn)
        conn.close()
        r = LocalZoteroReader(db_path=str(db_path))
        return r

    return _make


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_collection_key_filters_to_matching_items(reader_factory):
    r = reader_factory()
    try:
        items = r.get_items_with_text(collection_key="COLLXXX1")
    finally:
        r.close()

    keys = sorted(i.key for i in items)
    # X contains AA, BB, and DD (which is also in Y)
    assert keys == ["ITEMAA01", "ITEMBB02", "ITEMDD04"]


def test_collection_key_excludes_other_collections(reader_factory):
    r = reader_factory()
    try:
        items = r.get_items_with_text(collection_key="COLLYYY2")
    finally:
        r.close()

    keys = sorted(i.key for i in items)
    assert keys == ["ITEMCC03", "ITEMDD04"]


def test_no_row_duplication_for_multi_collection_item(reader_factory):
    """Item DD is in both X and Y — query on X should still return it once."""
    r = reader_factory()
    try:
        items = r.get_items_with_text(collection_key="COLLXXX1")
    finally:
        r.close()

    keys = [i.key for i in items]
    assert keys.count("ITEMDD04") == 1


def test_unknown_collection_key_returns_empty(reader_factory):
    r = reader_factory()
    try:
        items = r.get_items_with_text(collection_key="ZZZZZZZ9")
    finally:
        r.close()

    assert items == []


def test_default_behavior_unchanged_without_collection_key(reader_factory):
    """Unscoped call must return all 4 non-attachment items."""
    r = reader_factory()
    try:
        items = r.get_items_with_text()
    finally:
        r.close()

    keys = sorted(i.key for i in items)
    assert keys == ["ITEMAA01", "ITEMBB02", "ITEMCC03", "ITEMDD04"]
