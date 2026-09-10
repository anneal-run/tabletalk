"""Start a local llama.cpp server with OpenAI-compatible tool calling.

Downloads a prebuilt llama.cpp release on first run (Windows: CUDA 12.4 build if an NVIDIA
GPU is present, CPU build otherwise), then runs llama-server with --jinja so Qwen's chat
template emits native tool calls. Ctrl+C stops it.

    python scripts/llama_server.py --model models/Qwen2.5-7B-Instruct-Q4_K_M.gguf
"""

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

RELEASE = "b10809"
BASE = f"https://github.com/ggml-org/llama.cpp/releases/download/{RELEASE}"
ASSETS = {
    ("Windows", "cuda"): [
        f"llama-{RELEASE}-bin-win-cuda-12.4-x64.zip",
        "cudart-llama-bin-win-cuda-12.4-x64.zip",
    ],
    ("Windows", "cpu"): [f"llama-{RELEASE}-bin-win-cpu-x64.zip"],
    ("Linux", "cpu"): [f"llama-{RELEASE}-bin-ubuntu-x64.zip"],
}
ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "bin"


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


def find_server() -> Path | None:
    exe = "llama-server.exe" if platform.system() == "Windows" else "llama-server"
    hits = list(BIN.rglob(exe))
    return hits[0] if hits else None


def download(flavour: str) -> Path:
    key = (platform.system(), flavour)
    if key not in ASSETS:
        sys.exit(f"no prebuilt asset for {key}; build llama.cpp yourself and put "
                 f"llama-server in {BIN}")
    BIN.mkdir(exist_ok=True)
    for name in ASSETS[key]:
        target = BIN / name
        if not target.exists():
            print(f"downloading {name} ...")
            urllib.request.urlretrieve(f"{BASE}/{name}", target)
        with zipfile.ZipFile(target) as z:
            z.extractall(BIN)
    server = find_server()
    if server is None:
        sys.exit("download finished but llama-server was not found")
    # cudart dlls must sit next to the exe
    for dll in BIN.glob("*.dll"):
        dest = server.parent / dll.name
        if not dest.exists():
            shutil.copy(dll, dest)
    return server


def wait_healthy(port: int, timeout: float = 300) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1)
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=os.getenv("LLM_MODEL_PATH",
                    "models/Qwen2.5-7B-Instruct-Q4_K_M.gguf"))
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--ctx", type=int, default=8192)
    ap.add_argument("--ngl", type=int, default=99, help="GPU layers, 0 for CPU only")
    ap.add_argument("--cpu", action="store_true", help="force the CPU build")
    ap.add_argument("--alias", default="local", help="model name clients ask for")
    args = ap.parse_args()

    model = Path(args.model)
    if not model.exists():
        sys.exit(f"model not found: {model}. Run scripts/download_models.py first.")

    server = find_server()
    if server is None:
        flavour = "cpu" if args.cpu or not has_nvidia() else "cuda"
        server = download(flavour)

    cmd = [
        str(server), "-m", str(model), "--port", str(args.port), "--host", "127.0.0.1",
        "-c", str(args.ctx), "-ngl", str(0 if args.cpu else args.ngl),
        "--jinja", "-a", args.alias, "--log-disable",
    ]
    print("starting:", " ".join(cmd))
    proc = subprocess.Popen(cmd)
    try:
        if not wait_healthy(args.port):
            proc.terminate()
            sys.exit("server did not become healthy in time")
        print(f"llama-server ready on http://127.0.0.1:{args.port}/v1 (model alias '{args.alias}')")
        proc.wait()
    except KeyboardInterrupt:
        proc.terminate()


if __name__ == "__main__":
    main()
