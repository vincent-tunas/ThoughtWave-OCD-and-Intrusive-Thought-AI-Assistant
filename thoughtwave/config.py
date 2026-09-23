from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    hf_token: str
    openrouter_api_key: str
    openrouter_model: str
    hf_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    openrouter_timeout_seconds: float = 45.0
    openrouter_max_retries: int = 2
    similarity_threshold: float = 0.72
    top_k_related_messages: int = 5
    recent_history_limit: int = 10
    database_path: Path = Path("data/thoughtwave.db")
    app_url: str = "http://localhost:8501"

    @property
    def missing_required_values(self) -> list[str]:
        missing: list[str] = []
        if not self.hf_token:
            missing.append("HF_TOKEN")
        if not self.openrouter_api_key:
            missing.append("OPENROUTER_API_KEY")
        if not self.openrouter_model:
            missing.append("OPENROUTER_MODEL")
        return missing


def load_settings(env_file: str | Path | None = None) -> Settings:
    load_dotenv(dotenv_path=env_file, override=False)
    database_path = Path(os.getenv("DATABASE_PATH", "data/thoughtwave.db"))
    if not database_path.is_absolute():
        database_path = Path.cwd() / database_path

    return Settings(
        hf_token=os.getenv("HF_TOKEN", "").strip(),
        openrouter_api_key=os.getenv("OPENROUTER_API_KEY", "").strip(),
        openrouter_model=os.getenv("OPENROUTER_MODEL", "").strip(),
        hf_embedding_model=os.getenv(
            "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
        ).strip(),
        openrouter_base_url=os.getenv(
            "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
        ).rstrip("/"),
        openrouter_timeout_seconds=float(
            os.getenv("OPENROUTER_TIMEOUT_SECONDS", "45")
        ),
        openrouter_max_retries=int(os.getenv("OPENROUTER_MAX_RETRIES", "2")),
        similarity_threshold=float(os.getenv("SIMILARITY_THRESHOLD", "0.72")),
        top_k_related_messages=int(os.getenv("TOP_K_RELATED_MESSAGES", "5")),
        recent_history_limit=int(os.getenv("RECENT_HISTORY_LIMIT", "10")),
        database_path=database_path,
        app_url=os.getenv("APP_URL", "http://localhost:8501").strip(),
    )

