"""Download the GGUF used by scripts/llama_server.py into ./models.

    python scripts/download_models.py            # Qwen2.5-7B-Instruct Q4_K_M (~4.7 GB)
    python scripts/download_models.py --coder    # Qwen2.5-Coder-7B-Instruct Q4_K_M
    python scripts/download_models.py --small    # Qwen2.5-3B-Instruct Q4_K_M (~1.9 GB)
"""

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download

MODELS = {
    "default": ("bartowski/Qwen2.5-7B-Instruct-GGUF", "Qwen2.5-7B-Instruct-Q4_K_M.gguf"),
    "coder": ("bartowski/Qwen2.5-Coder-7B-Instruct-GGUF", "Qwen2.5-Coder-7B-Instruct-Q4_K_M.gguf"),
    "small": ("bartowski/Qwen2.5-3B-Instruct-GGUF", "Qwen2.5-3B-Instruct-Q4_K_M.gguf"),
}
ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coder", action="store_true")
    ap.add_argument("--small", action="store_true")
    args = ap.parse_args()
    key = "coder" if args.coder else "small" if args.small else "default"
    repo, filename = MODELS[key]
    target = ROOT / "models"
    target.mkdir(exist_ok=True)
    path = hf_hub_download(repo, filename, local_dir=target)
    print(f"ready: {path}\nset LLM_MODEL_PATH={Path(path).relative_to(ROOT)} in .env")


if __name__ == "__main__":
    main()
