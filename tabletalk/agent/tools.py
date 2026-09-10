"""Tool schemas the model sees. Execution happens in the graph."""

from langchain_core.tools import tool


@tool
def run_sql(sql: str) -> str:
    """Run one read-only SELECT query against the database and return the resulting rows."""
    raise NotImplementedError("executed by the graph")


@tool
def run_python(code: str) -> str:
    """Run Python (pandas, matplotlib, numpy) on the last query result, available as
    result.csv in the working directory. Save any chart to chart.png. Returns stdout."""
    raise NotImplementedError("executed by the graph")


@tool
def ask_user(question: str) -> str:
    """Ask the user one short clarifying question when the request is ambiguous. This ends
    your turn; the user's reply arrives as the next message."""
    raise NotImplementedError("executed by the graph")


TOOLS = [run_sql, run_python, ask_user]
