"""Run evals/behaviors.yaml against the live model. Prints a pass/fail table and writes
evals/results/behaviors.json. Exit code 1 if any case fails.

    python evals/run_behaviors.py [--only case_id]
"""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabletalk.agent.runner import Session  # noqa: E402

DB = ROOT / "data" / "chinook.db"


def db_digest() -> str:
    return hashlib.sha256(DB.read_bytes()).hexdigest()


def check(case: dict, turn, tools_used: list[str], db_unchanged: bool) -> list[str]:
    failures = []
    want = case.get("status")
    if want and turn.status != want:
        failures.append(f"status {turn.status} != {want}")
    for t in case.get("tools_include", []):
        if t not in tools_used:
            failures.append(f"tool {t} not used")
    for t in case.get("tools_exclude", []):
        if t in tools_used:
            failures.append(f"tool {t} used")
    any_of = case.get("answer_contains_any")
    if any_of and not any(s in turn.answer for s in any_of):
        failures.append(f"answer lacks any of {any_of}")
    for s in case.get("answer_not_contains", []):
        if s in turn.answer:
            failures.append(f"answer contains {s!r}")
    if case.get("chart") and not (turn.chart_path and Path(turn.chart_path).exists()):
        failures.append("no chart produced")
    if not db_unchanged:
        failures.append("DATABASE CHANGED")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only")
    args = ap.parse_args()

    cases = yaml.safe_load((ROOT / "evals" / "behaviors.yaml").read_text(encoding="utf-8"))
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    if not DB.exists():
        sys.exit("run scripts/get_chinook.py first")

    results = []
    for case in cases:
        before = db_digest()
        session = Session(str(DB))
        tools_used: list[str] = []
        t0 = time.time()
        turn = None
        error = None
        try:
            for q in case["turns"]:
                turn = session.ask(q)
                tools_used += [s.name for s in turn.trace.steps if s.kind == "tool"]
        except Exception as e:  # keep the suite going, count the case as failed
            error = f"{type(e).__name__}: {str(e)[:120]}"
        elapsed = time.time() - t0
        if turn is None or error:
            results.append({"id": case["id"], "pass": False, "failures": [error or "no turn"],
                            "seconds": round(elapsed, 1)})
            print(f"FAIL  {case['id']:<32} {'error':<9} {elapsed:5.1f}s  {error}")
            continue
        failures = check(case, turn, tools_used, db_digest() == before)
        ok = not failures
        results.append({
            "id": case["id"], "pass": ok, "failures": failures, "status": turn.status,
            "answer": turn.answer, "sql": turn.sql, "tools": tools_used,
            "llm_calls": turn.trace.llm_calls, "seconds": round(elapsed, 1),
        })
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {case['id']:<32} {turn.status:<9} {elapsed:5.1f}s  "
              f"{turn.trace.llm_calls} llm  {', '.join(failures)}")

    passed = sum(r["pass"] for r in results)
    print(f"\n{passed}/{len(results)} passed")
    out = ROOT / "evals" / "results"
    out.mkdir(exist_ok=True)
    name = f"behaviors_{args.only}.json" if args.only else "behaviors.json"
    (out / name).write_text(json.dumps(results, indent=2, ensure_ascii=False),
                            encoding="utf-8")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
