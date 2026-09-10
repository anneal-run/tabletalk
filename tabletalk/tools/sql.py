"""Read-only SQL execution with a parser-level guard.

Two independent layers: the guard rejects anything that is not a single SELECT before it
reaches SQLite, and the connection itself is opened read-only so a guard bug still cannot
write. Queries get a LIMIT and a wall-clock timeout so a bad join cannot hang the agent.
"""

import sqlite3
import time
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp

FORBIDDEN_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Command,
    exp.Pragma,
    exp.Transaction,
    exp.Commit,
    exp.Rollback,
)

ALLOWED_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)

# SQLite functions that reach outside the database.
FORBIDDEN_FUNCTIONS = {"load_extension", "readfile", "writefile", "fts3_tokenizer"}


class SqlGuardError(ValueError):
    """Raised when a query is refused before execution."""


@dataclass
class SqlResult:
    sql: str
    columns: list[str] = field(default_factory=list)
    rows: list[list] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    elapsed_ms: int = 0
    error: str | None = None

    def preview(self, n: int = 20) -> str:
        """Compact text form for the model: header line plus up to n rows."""
        if self.error:
            return f"ERROR: {self.error}"
        if not self.columns:
            return "(no columns)"
        lines = [" | ".join(self.columns)]
        for row in self.rows[:n]:
            lines.append(" | ".join("" if v is None else str(v) for v in row))
        if self.row_count > n:
            lines.append(f"... {self.row_count - n} more rows")
        if self.truncated:
            lines.append("(result capped by row limit)")
        return "\n".join(lines)


def guard_sql(sql: str, max_rows: int) -> str:
    """Validate a query and return the version that will actually run.

    Rules: exactly one statement, it must be a SELECT (or a set operation of SELECTs), no
    write/DDL/PRAGMA nodes anywhere in the tree, no functions that touch the filesystem, and
    a LIMIT no larger than max_rows.
    """
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise SqlGuardError(f"could not parse SQL: {e}") from e
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SqlGuardError("exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, ALLOWED_ROOTS):
        raise SqlGuardError(f"only SELECT queries are allowed, got {tree.key.upper()}")
    for node in tree.walk():
        if isinstance(node, FORBIDDEN_NODES):
            raise SqlGuardError(f"{node.key.upper()} is not allowed")
        if isinstance(node, exp.Func):
            name = (node.sql_name() or "").lower()
            if isinstance(node, exp.Anonymous):
                name = str(node.this).lower()
            if name in FORBIDDEN_FUNCTIONS:
                raise SqlGuardError(f"function {name}() is not allowed")
    limit = tree.args.get("limit")
    if limit is None:
        tree = tree.limit(max_rows)
    else:
        try:
            current = int(limit.expression.this)
        except (AttributeError, TypeError, ValueError):
            current = max_rows + 1
        if current > max_rows:
            tree = tree.limit(max_rows)
    return tree.sql(dialect="sqlite")


def open_readonly(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    conn.row_factory = None
    return conn


def run_sql(db_path: str, sql: str, max_rows: int = 200, timeout_s: float = 10.0) -> SqlResult:
    """Guard, then execute. Errors are returned in the result, not raised, so the agent
    can read them and repair the query."""
    try:
        safe_sql = guard_sql(sql, max_rows)
    except SqlGuardError as e:
        return SqlResult(sql=sql, error=str(e))

    deadline = time.monotonic() + timeout_s

    def watchdog():
        return 1 if time.monotonic() > deadline else 0

    start = time.monotonic()
    conn = open_readonly(db_path)
    try:
        conn.set_progress_handler(watchdog, 5000)
        cur = conn.execute(safe_sql)
        columns = [d[0] for d in cur.description] if cur.description else []
        rows = [list(r) for r in cur.fetchall()]
    except sqlite3.OperationalError as e:
        msg = str(e)
        if "interrupted" in msg:
            msg = f"query exceeded {timeout_s:.0f}s timeout"
        return SqlResult(sql=safe_sql, error=msg, elapsed_ms=_ms(start))
    except sqlite3.DatabaseError as e:
        return SqlResult(sql=safe_sql, error=str(e), elapsed_ms=_ms(start))
    finally:
        conn.close()

    # the guard caps LIMIT at max_rows, so hitting the cap means there may be more rows
    return SqlResult(
        sql=safe_sql,
        columns=columns,
        rows=rows,
        row_count=len(rows),
        truncated=len(rows) >= max_rows,
        elapsed_ms=_ms(start),
    )


def _ms(start: float) -> int:
    return int((time.monotonic() - start) * 1000)
