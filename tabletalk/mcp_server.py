"""MCP server: the agent as a tool for Claude Code, Claude Desktop or any MCP client.

    tabletalk serve-mcp data/chinook.db

Claude Code registration:
    claude mcp add tabletalk -- <path to .venv>/Scripts/tabletalk serve-mcp <db path>
"""

from mcp.server.fastmcp import FastMCP

from tabletalk.config import settings
from tabletalk.schema.index import load_schema, render_schema
from tabletalk.tools.sql import run_sql as _run_sql

_sessions: dict[str, object] = {}
_db_path = ""


def _session(thread_id: str):
    from tabletalk.agent.runner import Session

    if thread_id not in _sessions:
        _sessions[thread_id] = Session(_db_path, thread_id=thread_id)
    return _sessions[thread_id]


mcp = FastMCP("tabletalk")


@mcp.tool()
def ask_database(question: str, thread_id: str = "default") -> str:
    """Ask the database a question in plain language. The agent writes and runs the SQL
    itself. Reuse thread_id for follow-up questions. Returns the answer and the SQL used."""
    turn = _session(thread_id).ask(question)
    if turn.status == "clarify":
        return f"CLARIFICATION NEEDED: {turn.answer}"
    out = turn.answer
    if turn.sql and "```sql" not in out:
        out += f"\n\nSQL: {turn.sql}"
    if turn.chart_path:
        out += f"\nChart: {turn.chart_path}"
    return out


@mcp.tool()
def run_sql(sql: str) -> str:
    """Run one read-only SELECT directly and return the rows as text."""
    res = _run_sql(_db_path, sql, settings.max_rows, settings.sql_timeout_s)
    return res.preview(n=50)


@mcp.tool()
def describe_schema() -> str:
    """List tables, columns, foreign keys and sample rows."""
    return render_schema(load_schema(_db_path))


def serve(db_path: str) -> None:
    global _db_path
    _db_path = db_path
    mcp.run(transport="stdio")


if __name__ == "__main__":
    import sys

    serve(sys.argv[1])
