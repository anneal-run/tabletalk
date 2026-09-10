"""Command line interface."""

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from tabletalk.config import settings
from tabletalk.trace.store import Step, list_traces, read_trace

app = typer.Typer(add_completion=False, help="Ask a SQLite database questions in plain language.")
console = Console()


def _print_step(step: Step) -> None:
    if step.kind == "llm":
        calls = step.detail.get("tool_calls") or []
        what = f"-> {', '.join(calls)}" if calls else "-> answer"
        console.print(f"  [dim]llm[/dim] {what} [dim]({step.latency_ms} ms, "
                      f"{step.input_tokens}+{step.output_tokens} tok)[/dim]")
    elif step.kind == "tool":
        d = step.detail
        if step.name == "run_sql":
            sql = " ".join(str(d.get("sql", "")).split())
            if d.get("error"):
                console.print(f"  [red]sql[/red] {sql[:110]}\n      [red]{d['error']}[/red]")
            else:
                console.print(f"  [cyan]sql[/cyan] {sql[:110]} [dim]({d.get('rows')} rows, "
                              f"{step.latency_ms} ms)[/dim]")
        elif step.name == "run_python":
            status = "refused" if d.get("refused") else ("ok" if d.get("returncode") == 0
                                                          else "error")
            console.print(f"  [magenta]python[/magenta] {status} [dim]({step.latency_ms} ms)"
                          f"[/dim]")
    elif step.kind == "node" and step.detail.get("nudged"):
        console.print("  [yellow]verify[/yellow] answered without a query, asking again")


def _show_turn(turn) -> None:
    if turn.status == "clarify":
        console.print(Panel(turn.answer, title="question for you", border_style="yellow"))
        return
    console.print(Panel(Markdown(turn.answer), border_style="green" if turn.status == "answered"
                        else "red"))
    if turn.chart_path:
        console.print(f"chart: {turn.chart_path}")
        if os.name == "nt" and os.environ.get("TABLETALK_OPEN_CHARTS", "1") == "1":
            try:
                os.startfile(turn.chart_path)  # type: ignore[attr-defined]
            except OSError:
                pass
    t = turn.trace
    console.print(f"[dim]{t.llm_calls} llm calls, {t.tool_calls} tool calls, "
                  f"{t.input_tokens}+{t.output_tokens} tokens, {t.wall_ms/1000:.1f}s, "
                  f"run {t.run_id}[/dim]")


def _session(db: Path, thread: str | None, persist: bool):
    from tabletalk.agent.runner import Session

    if not db.exists():
        console.print(f"[red]database not found: {db}[/red]")
        raise typer.Exit(1)
    return Session(str(db), thread_id=thread, persist=persist)


@app.command()
def ask(db: Path, question: str, thread: str = typer.Option(None, help="thread id to continue"),
        persist: bool = typer.Option(False, help="keep conversation state on disk")):
    """Ask one question and exit."""
    s = _session(db, thread, persist)
    turn = s.ask(question, on_step=_print_step)
    _show_turn(turn)
    console.print(f"[dim]thread {s.thread_id}[/dim]")


@app.command()
def chat(db: Path, thread: str = typer.Option(None, help="thread id to continue"),
         persist: bool = typer.Option(False, help="keep conversation state on disk")):
    """Interactive session. Follow-up questions see the previous ones."""
    s = _session(db, thread, persist)
    console.print(f"[bold]tabletalk[/bold] on {db.name}  (thread {s.thread_id}; "
                  f"/new, /trace, /quit)")
    while True:
        try:
            q = console.input("[bold cyan]you>[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            continue
        if q in ("/quit", "/exit"):
            break
        if q == "/new":
            s = _session(db, None, persist)
            console.print(f"[dim]new thread {s.thread_id}[/dim]")
            continue
        if q == "/trace":
            _render_trace("latest")
            continue
        turn = s.ask(q, on_step=_print_step)
        _show_turn(turn)


@app.command()
def schema(db: Path):
    """Print the schema cards the model sees."""
    from tabletalk.schema.index import load_schema, render_schema

    console.print(render_schema(load_schema(str(db))))


def _render_trace(run_id: str) -> None:
    paths = list_traces(settings.trace_dir)
    if not paths:
        console.print("no traces yet")
        return
    path = paths[-1] if run_id == "latest" else Path(settings.trace_dir) / f"{run_id}.jsonl"
    if not path.exists():
        console.print(f"no trace {run_id}")
        return
    summary, steps = read_trace(path)
    console.print(Panel(f"[bold]{summary.get('question')}[/bold]\n"
                        f"status {summary.get('status')}, {summary.get('llm_calls')} llm calls, "
                        f"{summary.get('tool_calls')} tool calls, "
                        f"{summary.get('input_tokens')}+{summary.get('output_tokens')} tokens, "
                        f"{(summary.get('wall_ms') or 0)/1000:.1f}s",
                        title=summary.get("run_id")))
    table = Table(show_header=True, header_style="dim")
    table.add_column("#", justify="right")
    table.add_column("step")
    table.add_column("ms", justify="right")
    table.add_column("tokens", justify="right")
    table.add_column("detail")
    for i, st in enumerate(steps, 1):
        d = st.get("detail") or {}
        if st["kind"] == "tool" and st["name"] == "run_sql":
            detail = " ".join(str(d.get("sql", "")).split())[:80]
            if d.get("error"):
                detail += f"  ERROR {d['error'][:60]}"
        elif st["kind"] == "llm":
            detail = ", ".join(d.get("tool_calls") or []) or (d.get("content") or "")[:80]
        else:
            detail = ", ".join(f"{k}={v}" for k, v in d.items())[:80]
        tokens = f"{st.get('input_tokens', 0)}+{st.get('output_tokens', 0)}" \
            if st["kind"] == "llm" else ""
        table.add_row(str(i), f"{st['kind']}:{st['name']}", str(st["latency_ms"]), tokens, detail)
    console.print(table)


@app.command()
def trace(run_id: str = typer.Argument("latest")):
    """Show the steps, latency and tokens of a run."""
    _render_trace(run_id)


@app.command(name="serve-mcp")
def serve_mcp(db: Path):
    """Expose the agent as an MCP server on stdio."""
    from tabletalk.mcp_server import serve

    serve(str(db))
