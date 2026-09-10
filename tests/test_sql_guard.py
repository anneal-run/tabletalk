"""SQL guard rules."""

import sqlite3

import pytest

from tabletalk.tools.sql import SqlGuardError, guard_sql, run_sql


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "shop.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, country TEXT);
        CREATE TABLE invoices (id INTEGER PRIMARY KEY, customer_id INTEGER, total REAL);
        INSERT INTO customers VALUES (1, 'Ana', 'PT'), (2, 'Bo', 'DE'), (3, 'Cy', 'DE');
        INSERT INTO invoices VALUES (1, 1, 10.0), (2, 2, 5.5), (3, 2, 4.5), (4, 3, 100.0);
        """
    )
    conn.commit()
    conn.close()
    return str(path)


def test_plain_select_runs_and_gets_a_limit(db):
    res = run_sql(db, "SELECT name FROM customers ORDER BY id", max_rows=2)
    assert res.error is None
    assert res.rows == [["Ana"], ["Bo"]]
    assert res.truncated is True
    assert "LIMIT 2" in res.sql


def test_existing_limit_is_kept_when_small_enough():
    assert guard_sql("SELECT 1 LIMIT 5", max_rows=10).endswith("LIMIT 5")


def test_existing_limit_is_capped_when_too_large():
    assert guard_sql("SELECT 1 LIMIT 5000", max_rows=10).endswith("LIMIT 10")


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM customers",
        "UPDATE customers SET name = 'x'",
        "INSERT INTO customers VALUES (9, 'z', 'FR')",
        "DROP TABLE customers",
        "CREATE TABLE t (x INT)",
        "ALTER TABLE customers ADD COLUMN x INT",
        "PRAGMA table_info(customers)",
        "SELECT 1; DELETE FROM customers",
        "SELECT load_extension('evil')",
    ],
)
def test_writes_and_escapes_are_refused(sql):
    with pytest.raises(SqlGuardError):
        guard_sql(sql, max_rows=10)


def test_write_hidden_in_cte_is_refused():
    # sqlite does not allow this either, but the guard must catch it before sqlite sees it
    with pytest.raises(SqlGuardError):
        guard_sql("WITH x AS (DELETE FROM customers RETURNING id) SELECT * FROM x", 10)


def test_set_operations_and_ctes_are_allowed(db):
    sql = """
        WITH de AS (SELECT id FROM customers WHERE country = 'DE')
        SELECT id FROM de UNION SELECT id FROM customers WHERE country = 'PT'
    """
    res = run_sql(db, sql)
    assert res.error is None
    assert sorted(r[0] for r in res.rows) == [1, 2, 3]


def test_connection_is_read_only_even_if_guard_is_bypassed(db):
    from tabletalk.tools.sql import open_readonly

    conn = open_readonly(db)
    with pytest.raises(sqlite3.OperationalError):
        conn.execute("DELETE FROM customers")
    conn.close()


def test_sql_errors_come_back_as_results_not_exceptions(db):
    res = run_sql(db, "SELECT nope FROM customers")
    assert res.error is not None
    assert "nope" in res.error
    assert res.rows == []


def test_timeout_interrupts_long_queries(db):
    # cross join of a recursive counter: effectively infinite
    sql = """
        WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c)
        SELECT count(*) FROM c a, c b
    """
    res = run_sql(db, sql, timeout_s=0.2)
    assert res.error is not None
    assert "timeout" in res.error


def test_preview_is_compact(db):
    res = run_sql(db, "SELECT name, country FROM customers ORDER BY id")
    text = res.preview(n=2)
    assert text.splitlines()[0] == "name | country"
    assert "Ana | PT" in text
    assert "1 more rows" in text
