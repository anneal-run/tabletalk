"""Settings read from the environment (.env is loaded once at import)."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Settings:
    base_url: str = os.getenv("TABLETALK_BASE_URL", "http://127.0.0.1:8080/v1")
    model: str = os.getenv("TABLETALK_MODEL", "local")
    api_key: str = os.getenv("TABLETALK_API_KEY", "sk-local")
    max_rows: int = int(os.getenv("TABLETALK_MAX_ROWS", "200"))
    sql_timeout_s: float = float(os.getenv("TABLETALK_SQL_TIMEOUT", "10"))
    max_repairs: int = int(os.getenv("TABLETALK_MAX_REPAIRS", "3"))
    schema_token_budget: int = int(os.getenv("TABLETALK_SCHEMA_BUDGET", "2000"))
    sandbox_timeout_s: float = float(os.getenv("TABLETALK_SANDBOX_TIMEOUT", "20"))
    trace_dir: str = os.getenv("TABLETALK_TRACE_DIR", "traces")
    chart_dir: str = os.getenv("TABLETALK_CHART_DIR", "charts")
    checkpoint_path: str = os.getenv("TABLETALK_CHECKPOINTS", "checkpoints/threads.sqlite")


settings = Settings()
