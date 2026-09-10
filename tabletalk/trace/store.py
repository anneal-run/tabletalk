"""Run traces as JSONL: one file per run, one line per step.

A step is an LLM call or a tool call with its latency and, for LLM calls, prompt and
completion tokens as reported by the server. `tabletalk trace <run_id>` renders a file.
"""

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Step:
    kind: str  # "llm" | "tool" | "node"
    name: str
    latency_ms: int
    input_tokens: int = 0
    output_tokens: int = 0
    detail: dict = field(default_factory=dict)
    t: float = field(default_factory=time.time)


@dataclass
class RunTrace:
    run_id: str
    thread_id: str
    question: str
    steps: list[Step] = field(default_factory=list)
    started: float = field(default_factory=time.time)
    status: str = "running"
    answer: str = ""

    @property
    def llm_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "llm")

    @property
    def tool_calls(self) -> int:
        return sum(1 for s in self.steps if s.kind == "tool")

    @property
    def input_tokens(self) -> int:
        return sum(s.input_tokens for s in self.steps)

    @property
    def output_tokens(self) -> int:
        return sum(s.output_tokens for s in self.steps)

    @property
    def wall_ms(self) -> int:
        return int((time.time() - self.started) * 1000)

    def summary(self) -> dict:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "question": self.question,
            "status": self.status,
            "llm_calls": self.llm_calls,
            "tool_calls": self.tool_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "wall_ms": self.wall_ms,
            "answer": self.answer,
        }


def new_run_id() -> str:
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def write_trace(trace: RunTrace, trace_dir: str) -> Path:
    path = Path(trace_dir) / f"{trace.run_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"summary": trace.summary()}) + "\n")
        for step in trace.steps:
            f.write(json.dumps({"step": asdict(step)}, default=str) + "\n")
    return path


def read_trace(path: Path) -> tuple[dict, list[dict]]:
    summary, steps = {}, []
    with path.open(encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "summary" in obj:
                summary = obj["summary"]
            elif "step" in obj:
                steps.append(obj["step"])
    return summary, steps


def list_traces(trace_dir: str) -> list[Path]:
    return sorted(Path(trace_dir).glob("*.jsonl"))
