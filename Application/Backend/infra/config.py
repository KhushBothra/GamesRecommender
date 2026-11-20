"""Central configuration for the local backend."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    use_groq: bool
    groq_api_key: str | None
    groq_model: str
    ollama_base_url: str
    ollama_chat_url: str
    ollama_model: str
    ollama_timeout_seconds: int
    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str
    qdrant_api_key: str | None
    games_descriptors_path: Path


def get_settings() -> Settings:
    """Read settings from environment variables, with local-dev defaults."""
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    return Settings(
        use_groq=_env_bool("USE_GROQ", True),
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip() or None,
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        ollama_base_url=base_url,
        ollama_chat_url=os.getenv("OLLAMA_CHAT_URL", f"{base_url}/api/chat").strip(),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b").strip(),
        ollama_timeout_seconds=int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180")),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost").strip(),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection_name=os.getenv(
            "QDRANT_COLLECTION_NAME", "game_embeddings"
        ).strip(),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
        games_descriptors_path=Path(
            os.getenv(
                "GAMES_DESCRIPTORS_PATH",
                str(BACKEND_DIR.parent / "Data Prep" / "games_descriptors.csv"),
            )
        ).expanduser(),
    )


settings = get_settings()
