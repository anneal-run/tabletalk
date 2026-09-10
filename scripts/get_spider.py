"""Fetch the Spider dev set (questions, gold SQL, 166 SQLite databases) for evals/spider.py.

Spider is distributed as a ~200 MB zip on Google Drive (CC BY-SA 4.0). Requires `pip install
gdown`. The zip contains macOS resource-fork entries that trip some unzip tools, so they are
skipped here.

    python scripts/get_spider.py
"""

import subprocess
import sys
import zipfile
from pathlib import Path

FILE_ID = "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J"
ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data" / "spider"
ZIP = DEST / "spider_data.zip"


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    if not ZIP.exists():
        subprocess.run([sys.executable, "-m", "gdown", FILE_ID, "-O", str(ZIP)], check=True)
    with zipfile.ZipFile(ZIP) as z:
        members = [n for n in z.namelist()
                   if "__MACOSX" not in n and not Path(n).name.startswith("._")]
        z.extractall(DEST, members=members)
    dbs = list((DEST / "spider_data" / "database").glob("*/*.sqlite"))
    print(f"ready: {DEST / 'spider_data'} ({len(dbs)} databases)")


if __name__ == "__main__":
    main()
