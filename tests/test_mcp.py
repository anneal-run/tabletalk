"""The MCP server speaks the protocol: tools are listed and the model-free ones work."""

import asyncio
import sqlite3
import sys

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


def _db(tmp_path):
    path = tmp_path / "t.db"
    conn = sqlite3.connect(path)
    conn.executescript("CREATE TABLE t (x INT); INSERT INTO t VALUES (1), (2);")
    conn.commit()
    conn.close()
    return str(path)


async def _exercise(db_path: str):
    params = StdioServerParameters(command=sys.executable,
                                   args=["-m", "tabletalk.mcp_server", db_path])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name for t in (await session.list_tools()).tools}
            schema = await session.call_tool("describe_schema", {})
            rows = await session.call_tool("run_sql", {"sql": "SELECT sum(x) AS s FROM t"})
            refused = await session.call_tool("run_sql", {"sql": "DELETE FROM t"})
            return tools, schema.content[0].text, rows.content[0].text, refused.content[0].text


def test_mcp_lists_tools_and_answers_model_free_calls(tmp_path):
    tools, schema, rows, refused = asyncio.run(_exercise(_db(tmp_path)))
    assert {"ask_database", "run_sql", "describe_schema"} <= tools
    assert "TABLE t" in schema
    assert rows.splitlines()[:2] == ["s", "3"]
    assert "only SELECT" in refused
