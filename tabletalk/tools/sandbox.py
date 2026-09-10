"""Run model-written Python in a separate process.

Static checks first (module allowlist, no dangerous builtins, no URLs), then a subprocess in
isolated mode with a wall-clock timeout, a fresh temp directory as cwd, an empty environment
and capped output. The last query result is given to the script as result.csv; a chart is
expected at chart.png. Not a security boundary; no memory limit on Windows.
"""

import ast
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ALLOWED_MODULES = {
    "pandas", "numpy", "matplotlib", "math", "statistics", "json", "csv", "datetime",
    "collections", "itertools", "re", "functools", "textwrap", "decimal", "fractions",
}
BANNED_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint", "globals",
    "locals", "getattr", "setattr", "delattr", "vars", "memoryview",
}
URL_RE = re.compile(r"https?://", re.IGNORECASE)
MAX_OUTPUT = 4000


class SandboxError(ValueError):
    """Raised when code is refused before execution."""


@dataclass
class SandboxResult:
    stdout: str
    stderr: str
    chart_path: str | None
    timed_out: bool
    returncode: int


def check_code(code: str) -> None:
    if URL_RE.search(code):
        raise SandboxError("network access is not allowed (URL found in code)")
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        raise SandboxError(f"syntax error: {e}") from e
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_MODULES:
                    raise SandboxError(f"import of {root} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if root not in ALLOWED_MODULES:
                raise SandboxError(f"import from {root} is not allowed")
        elif isinstance(node, ast.Name) and node.id in BANNED_NAMES:
            raise SandboxError(f"{node.id}() is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise SandboxError("dunder attribute access is not allowed")


PRELUDE = """
import matplotlib
matplotlib.use("Agg")
"""


def run_python(code: str, result_csv: str | None, timeout_s: float = 20.0,
               chart_out: str | None = None) -> SandboxResult:
    check_code(code)
    workdir = Path(tempfile.mkdtemp(prefix="tabletalk_"))
    try:
        if result_csv is not None:
            (workdir / "result.csv").write_text(result_csv, encoding="utf-8")
        script = workdir / "script.py"
        script.write_text(PRELUDE + code, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "MPLBACKEND": "Agg",
               "MPLCONFIGDIR": str(workdir / "mpl")}
        if os.name == "nt":
            env["SYSTEMROOT"] = os.environ.get("SYSTEMROOT", r"C:\Windows")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                cwd=workdir,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            stdout, stderr, rc, timed_out = proc.stdout, proc.stderr, proc.returncode, False
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr = f"killed after {timeout_s:.0f}s"
            rc, timed_out = -1, True
        chart_path = None
        produced = workdir / "chart.png"
        if produced.exists() and chart_out:
            Path(chart_out).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(produced, chart_out)
            chart_path = chart_out
        return SandboxResult(
            stdout=stdout[:MAX_OUTPUT],
            stderr=stderr[-MAX_OUTPUT:],
            chart_path=chart_path,
            timed_out=timed_out,
            returncode=rc,
        )
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
