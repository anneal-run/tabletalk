"""Aggregate traces into the cost/latency table used in the README.

    python evals/report.py [--price-in 0.15 --price-out 0.60]   (USD per 1M tokens)

Local inference costs nothing per token; the price flags estimate what the same runs would
cost on a hosted model.
"""

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tabletalk.config import settings  # noqa: E402
from tabletalk.trace.store import list_traces, read_trace  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--price-in", type=float, default=0.15)
    ap.add_argument("--price-out", type=float, default=0.60)
    args = ap.parse_args()

    runs = [read_trace(p)[0] for p in list_traces(settings.trace_dir)]
    runs = [r for r in runs if r.get("status") in ("answered", "clarify", "failed")]
    if not runs:
        print("no traces")
        return 0

    def col(key):
        return [r.get(key, 0) or 0 for r in runs]

    def line(label, values, fmt="{:.1f}"):
        print(f"| {label:<22} | {fmt.format(statistics.mean(values)):>8} | "
              f"{fmt.format(statistics.median(values)):>8} | {fmt.format(max(values)):>8} |")

    print(f"{len(runs)} runs from {settings.trace_dir}\n")
    print("| metric                 |     mean |   median |      max |")
    print("|------------------------|---------:|---------:|---------:|")
    line("llm calls", col("llm_calls"))
    line("tool calls", col("tool_calls"))
    line("prompt tokens", col("input_tokens"), "{:.0f}")
    line("completion tokens", col("output_tokens"), "{:.0f}")
    line("wall seconds", [v / 1000 for v in col("wall_ms")])
    cost = [(i * args.price_in + o * args.price_out) / 1e6
            for i, o in zip(col("input_tokens"), col("output_tokens"))]
    line("est. hosted cost USD", cost, "{:.4f}")
    by_status = {}
    for r in runs:
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
    print("\nstatus:", ", ".join(f"{k} {v}" for k, v in sorted(by_status.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
