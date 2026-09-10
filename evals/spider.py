"""Execution accuracy on a fixed sample of the Spider dev set.

Spider ships as a zip on Google Drive (see README). Expected layout after unzipping into
data/spider: spider_data/dev.json and spider_data/database/<db_id>/<db_id>.sqlite.

A prediction is correct when the model's final SQL returns the same rows as the gold SQL
on the same database (order-sensitive only if the gold query has ORDER BY). Runs where the
agent asked for clarification or produced no SQL count as wrong and are reported separately.

    python evals/spider.py --n 100 --seed 7
"""

import argparse
import json
import random
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabletalk.agent.runner import Session  # noqa: E402

SPIDER = ROOT / "data" / "spider" / "spider_data"


def find_db(db_id: str) -> Path | None:
    for cand in (SPIDER / "database" / db_id / f"{db_id}.sqlite",
                 SPIDER / "test_database" / db_id / f"{db_id}.sqlite"):
        if cand.exists():
            return cand
    return None


def execute(db: Path, sql: str, timeout_s: float = 30.0):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 5000)
    try:
        rows = conn.execute(sql).fetchall()
    finally:
        conn.close()
    return [tuple(_norm(v) for v in r) for r in rows]


def _norm(v):
    if isinstance(v, float):
        return round(v, 3)
    return v


def same_result(gold_rows, pred_rows, ordered: bool) -> bool:
    if ordered:
        return gold_rows == pred_rows
    return sorted(map(str, gold_rows)) == sorted(map(str, pred_rows))


def last_raw_sql(turn) -> str | None:
    """The model's own SQL from the last successful run_sql call (before the guard's LIMIT)."""
    for step in reversed(turn.trace.steps):
        if step.kind == "tool" and step.name == "run_sql" and not step.detail.get("error"):
            return step.detail.get("sql")
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    dev_path = SPIDER / "dev.json"
    if not dev_path.exists():
        sys.exit(f"missing {dev_path}; download Spider first (README, Evals section)")
    dev = json.loads(dev_path.read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    sample = rng.sample(dev, min(args.n, len(dev)))

    results = []
    correct = 0
    t_start = time.time()
    for i, ex in enumerate(sample, 1):
        db = find_db(ex["db_id"])
        if db is None:
            results.append({"db_id": ex["db_id"], "question": ex["question"],
                            "outcome": "no_db"})
            continue
        session = Session(str(db), write_traces=False)
        t0 = time.time()
        try:
            turn = session.ask(ex["question"])
        except Exception as e:  # server hiccup, count as wrong but keep going
            results.append({"db_id": ex["db_id"], "question": ex["question"],
                            "outcome": "exception", "error": str(e)[:200]})
            print(f"{i:3d} EXC   {ex['db_id']:<24} {str(e)[:80]}")
            continue
        elapsed = time.time() - t0
        pred_sql = last_raw_sql(turn)
        outcome = "wrong"
        if turn.status == "clarify":
            outcome = "clarify"
        elif pred_sql is None:
            outcome = "no_sql"
        else:
            try:
                gold_rows = execute(db, ex["query"])
                pred_rows = execute(db, pred_sql)
                ordered = "order by" in ex["query"].lower()
                if same_result(gold_rows, pred_rows, ordered):
                    outcome = "correct"
                    correct += 1
            except sqlite3.Error as e:
                outcome = "exec_error"
                pred_sql = f"{pred_sql}  -- {e}"
        results.append({
            "db_id": ex["db_id"], "question": ex["question"], "gold": ex["query"],
            "pred": pred_sql, "outcome": outcome, "status": turn.status,
            "llm_calls": turn.trace.llm_calls, "tool_calls": turn.trace.tool_calls,
            "input_tokens": turn.trace.input_tokens, "output_tokens": turn.trace.output_tokens,
            "seconds": round(elapsed, 1),
        })
        print(f"{i:3d} {outcome:<10} {ex['db_id']:<24} {elapsed:5.1f}s "
              f"{turn.trace.llm_calls} llm  {ex['question'][:60]}")

    n = len(results)
    counts = {}
    for r in results:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    scored = [r for r in results if "seconds" in r]
    summary = {
        "n": n, "correct": correct, "execution_accuracy": round(correct / n, 3) if n else 0,
        "outcomes": counts, "seed": args.seed, "tag": args.tag,
        "mean_seconds": round(sum(r["seconds"] for r in scored) / max(len(scored), 1), 1),
        "mean_llm_calls": round(sum(r["llm_calls"] for r in scored) / max(len(scored), 1), 2),
        "mean_input_tokens": int(sum(r["input_tokens"] for r in scored) / max(len(scored), 1)),
        "mean_output_tokens": int(sum(r["output_tokens"] for r in scored) / max(len(scored), 1)),
        "total_minutes": round((time.time() - t_start) / 60, 1),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    name = f"spider_n{args.n}_s{args.seed}{('_' + args.tag) if args.tag else ''}.json"
    (out / name).write_text(json.dumps({"summary": summary, "results": results}, indent=2,
                                       ensure_ascii=False), encoding="utf-8")
    print(f"written {out / name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
