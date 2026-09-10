"""Schema cards and the size threshold that decides whether retrieval runs at all."""

import sqlite3

import pytest

from tabletalk.schema.index import load_schema, render_schema, select_tables
from tests.fakes import keyword_embedder


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE artists (id INTEGER PRIMARY KEY, name TEXT);
        CREATE TABLE albums (id INTEGER PRIMARY KEY, title TEXT,
                             artist_id INTEGER REFERENCES artists(id));
        CREATE TABLE invoices (id INTEGER PRIMARY KEY, total REAL);
        CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, city TEXT);
        INSERT INTO artists VALUES (1, 'AC/DC'), (2, 'Accept');
        INSERT INTO albums VALUES (1, 'Let There Be Rock', 1);
        INSERT INTO invoices VALUES (1, 9.9);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_cards_carry_columns_fks_samples_and_counts(db):
    cards = {c.name: c for c in load_schema(db)}
    albums = cards["albums"]
    assert [c[0] for c in albums.columns] == ["id", "title", "artist_id"]
    assert albums.columns[0][2] is True, "id is the primary key"
    assert albums.foreign_keys == [("artist_id", "artists", "id")]
    assert albums.sample_rows == [[1, "Let There Be Rock", 1]]
    assert cards["artists"].row_count == 2
    text = render_schema(load_schema(db))
    assert "FK albums.artist_id -> artists.id" in text
    assert "AC/DC" in text


def test_small_schema_is_passed_whole_without_embedding(db):
    def must_not_run(texts):
        raise AssertionError("embedder should not be called for a small schema")

    sel = select_tables(load_schema(db), "how many albums", 100_000, embed=must_not_run)
    assert sel.retrieved is False
    assert len(sel.cards) == 4


def test_large_schema_retrieves_relevant_tables_and_their_fk_neighbours(db):
    cards = load_schema(db)
    embed = keyword_embedder(["album", "artist", "invoice", "employee"])
    sel = select_tables(cards, "which albums exist", token_budget=80, embed=embed)
    names = [c.name for c in sel.cards]
    assert sel.retrieved is True
    assert "albums" in names
    assert "artists" in names, "FK neighbour of albums is pulled in for the join"
    assert "employees" not in names
