"""The agent as a LangGraph state machine.

    prepare -> agent -> tools -> agent -> ... -> verify -> END
                  '-> clarify -> END

prepare builds or extends the system prompt with the schema. agent is one model call with
tools bound. tools executes run_sql / run_python, counts repairs and cuts the model off when
the repair budget is spent. verify pushes back once if the model answered with numbers
without querying. clarify ends the turn with a question for the user.
"""

import csv
import io
import re
import time
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    RemoveMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from tabletalk.agent import prompts
from tabletalk.agent.tools import TOOLS
from tabletalk.config import settings
from tabletalk.schema.index import load_schema, render_schema, select_tables
from tabletalk.tools.sandbox import SandboxError, run_python
from tabletalk.tools.sql import run_sql
from tabletalk.trace.store import RunTrace, Step


class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]
    db_path: str
    question: str
    schema_shown: list[str]
    # per-turn fields, reset in prepare
    repairs: int
    sql_ok: bool
    nudged: bool
    last_sql: str | None
    last_result_csv: str | None
    status: str
    answer: str
    chart_path: str | None


_schema_cache: dict[str, list] = {}


def _cards(db_path: str):
    if db_path not in _schema_cache:
        _schema_cache[db_path] = load_schema(db_path)
    return _schema_cache[db_path]


def _tracer(config: RunnableConfig) -> RunTrace | None:
    return (config.get("configurable") or {}).get("tracer")


def _record(config: RunnableConfig, step: Step) -> None:
    tracer = _tracer(config)
    if tracer is not None:
        tracer.steps.append(step)
        cb = getattr(tracer, "on_step", None)
        if cb:
            cb(step)


def _to_csv(columns: list[str], rows: list[list]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(columns)
    w.writerows(rows)
    return buf.getvalue()


def build_graph(model, checkpointer=None, embed=None):
    """model: any LangChain chat model with bind_tools. embed: optional embedder override
    used by schema retrieval (tests pass a fake)."""

    def prepare(state: AgentState, config: RunnableConfig) -> dict:
        t0 = time.monotonic()
        cards = _cards(state["db_path"])
        selection = select_tables(cards, state["question"], settings.schema_token_budget, embed)
        shown = list(state.get("schema_shown") or [])
        update: dict = {
            "repairs": 0,
            "sql_ok": False,
            "nudged": False,
            "last_sql": None,
            "last_result_csv": None,
            "status": "running",
            "answer": "",
            "chart_path": None,
        }
        messages = state.get("messages") or []
        has_system = bool(messages) and isinstance(messages[0], SystemMessage)
        if not has_system:
            note = (
                prompts.SCHEMA_NOTE_PARTIAL.format(shown=len(selection.cards),
                                                   total=selection.total_tables)
                if selection.retrieved
                else prompts.SCHEMA_NOTE_FULL
            )
            system = SystemMessage(
                prompts.SYSTEM_PROMPT.format(schema=selection.text, schema_note=note)
            )
            # the reducer only appends, so re-add the question after the system message
            humans = [m for m in messages if isinstance(m, HumanMessage)]
            update["messages"] = (
                [RemoveMessage(id=m.id) for m in humans if m.id]
                + [system]
                + [HumanMessage(m.content) for m in humans]
            )
            shown = [c.name for c in selection.cards]
        else:
            new_cards = [c for c in selection.cards if c.name not in shown]
            if new_cards:
                last = messages[-1]
                folded = HumanMessage(
                    prompts.FOLLOWUP_SCHEMA.format(schema=render_schema(new_cards),
                                                   question=last.content),
                    id=last.id,
                )
                update["messages"] = [folded]
                shown = shown + [c.name for c in new_cards]
        update["schema_shown"] = shown
        _record(config, Step("node", "prepare", int((time.monotonic() - t0) * 1000),
                             detail={"tables_shown": len(shown),
                                     "retrieved": selection.retrieved}))
        return update

    def agent(state: AgentState, config: RunnableConfig) -> dict:
        exhausted = state.get("repairs", 0) >= settings.max_repairs
        bound = model if exhausted else model.bind_tools(TOOLS)
        t0 = time.monotonic()
        msg = bound.invoke(state["messages"])
        usage = getattr(msg, "usage_metadata", None) or {}
        _record(config, Step(
            "llm", "agent", int((time.monotonic() - t0) * 1000),
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
            detail={"tool_calls": [c["name"] for c in getattr(msg, "tool_calls", [])],
                    "content": (msg.content or "")[:300]},
        ))
        return {"messages": [msg]}

    def tools(state: AgentState, config: RunnableConfig) -> dict:
        last: AIMessage = state["messages"][-1]
        out: list = []
        repairs = state.get("repairs", 0)
        sql_ok = state.get("sql_ok", False)
        last_sql = state.get("last_sql")
        last_csv = state.get("last_result_csv")
        chart_path = state.get("chart_path")
        tracer = _tracer(config)
        run_id = tracer.run_id if tracer else "run"

        for call in last.tool_calls:
            name, args, call_id = call["name"], call["args"], call["id"]
            t0 = time.monotonic()
            if name == "run_sql":
                sql = str(args.get("sql", ""))
                res = run_sql(state["db_path"], sql, settings.max_rows, settings.sql_timeout_s)
                if res.error:
                    repairs += 1
                    content = f"ERROR: {res.error}"
                    if repairs >= settings.max_repairs:
                        content += "\n" + prompts.REPAIR_EXHAUSTED
                else:
                    sql_ok = True
                    last_sql = res.sql
                    last_csv = _to_csv(res.columns, res.rows)
                    content = res.preview(n=30) + f"\n({res.row_count} rows, {res.elapsed_ms} ms)"
                detail = {"sql": sql, "error": res.error, "rows": res.row_count}
            elif name == "run_python":
                code = str(args.get("code", ""))
                target = f"{settings.chart_dir}/{run_id}.png"
                try:
                    sb = run_python(code, last_csv, settings.sandbox_timeout_s, chart_out=target)
                    parts = []
                    if sb.stdout.strip():
                        parts.append(sb.stdout.strip())
                    if sb.chart_path:
                        chart_path = sb.chart_path
                        parts.append(f"chart saved: {sb.chart_path}")
                    if sb.returncode != 0:
                        parts.append(f"ERROR: {sb.stderr.strip()[-800:]}")
                    content = "\n".join(parts) or "(no output)"
                    detail = {"returncode": sb.returncode, "chart": sb.chart_path,
                              "timed_out": sb.timed_out}
                except SandboxError as e:
                    content = f"ERROR: {e}"
                    detail = {"refused": str(e)}
            else:
                content = f"ERROR: unknown tool {name}"
                detail = {}
            _record(config, Step("tool", name, int((time.monotonic() - t0) * 1000),
                                 detail=detail))
            out.append(ToolMessage(content=content, tool_call_id=call_id, name=name))

        return {"messages": out, "repairs": repairs, "sql_ok": sql_ok, "last_sql": last_sql,
                "last_result_csv": last_csv, "chart_path": chart_path}

    def clarify(state: AgentState, config: RunnableConfig) -> dict:
        last: AIMessage = state["messages"][-1]
        call = next(c for c in last.tool_calls if c["name"] == "ask_user")
        question = str(call["args"].get("question", "")).strip() or "Could you clarify?"
        # every tool call needs a tool message, or the next model call is rejected
        out = [ToolMessage(content="Question sent to the user.", tool_call_id=c["id"],
                           name=c["name"]) for c in last.tool_calls]
        _record(config, Step("node", "clarify", 0, detail={"question": question}))
        return {"messages": out, "status": "clarify", "answer": question}

    def verify(state: AgentState, config: RunnableConfig) -> dict:
        last: AIMessage = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        has_numbers = bool(re.search(r"\d", text))
        if has_numbers and not state.get("sql_ok") and not state.get("nudged") \
                and state.get("repairs", 0) == 0:
            _record(config, Step("node", "verify", 0, detail={"nudged": True}))
            return {"messages": [HumanMessage(prompts.NUDGE)], "nudged": True}
        status = "answered"
        if state.get("repairs", 0) >= settings.max_repairs and not state.get("sql_ok"):
            status = "failed"
        if text.count("```") % 2 == 1:  # small models often forget to close the fence
            text = text.rstrip() + "\n```"
        _record(config, Step("node", "verify", 0, detail={"status": status}))
        return {"status": status, "answer": text}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if any(c["name"] == "ask_user" for c in calls):
            return "clarify"
        if calls:
            return "tools"
        return "verify"

    def route_after_verify(state: AgentState) -> str:
        return "agent" if state.get("status") == "running" else END

    g = StateGraph(AgentState)
    g.add_node("prepare", prepare)
    g.add_node("agent", agent)
    g.add_node("tools", tools)
    g.add_node("clarify", clarify)
    g.add_node("verify", verify)
    g.add_edge(START, "prepare")
    g.add_edge("prepare", "agent")
    g.add_conditional_edges("agent", route_after_agent,
                            {"tools": "tools", "clarify": "clarify", "verify": "verify"})
    g.add_edge("tools", "agent")
    g.add_edge("clarify", END)
    g.add_conditional_edges("verify", route_after_verify, {"agent": "agent", END: END})
    return g.compile(checkpointer=checkpointer)
