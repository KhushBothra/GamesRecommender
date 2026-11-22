"""Central configuration for the local backend."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")


def _groq_api_keys() -> tuple[str, ...]:
    """Read numbered Groq keys in order, then the legacy unnumbered key."""
    numbered: list[tuple[int, str]] = []
    for name, value in os.environ.items():
        match = re.fullmatch(r"GROQ_API_KEY(\d+)", name)
        if match and value.strip():
            numbered.append((int(match.group(1)), value.strip()))

    ordered = [value for _, value in sorted(numbered)]
    legacy = os.getenv("GROQ_API_KEY", "").strip()
    if legacy:
        ordered.append(legacy)

    # Avoid retrying the same credential if it was copied under multiple names.
    unique: list[str] = []
    seen: set[str] = set()
    for value in ordered:
        if value not in seen:
            unique.append(value)
            seen.add(value)
    return tuple(unique)


@dataclass(frozen=True)
class Settings:
    groq_api_keys: tuple[str, ...]
    groq_model: str
    qdrant_host: str
    qdrant_port: int
    qdrant_collection_name: str
    qdrant_api_key: str | None
    games_descriptors_path: Path


def get_settings() -> Settings:
    """Read settings from environment variables, with local-dev defaults."""
    return Settings(
        groq_api_keys=_groq_api_keys(),
        groq_model=os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip(),
        qdrant_host=os.getenv("QDRANT_HOST", "localhost").strip(),
        qdrant_port=int(os.getenv("QDRANT_PORT", "6333")),
        qdrant_collection_name=os.getenv(
            "QDRANT_COLLECTION_NAME", "game_embeddings"
        ).strip(),
        qdrant_api_key=os.getenv("QDRANT_API_KEY", "").strip() or None,
        games_descriptors_path=Path(
            os.getenv(
                "GAMES_DESCRIPTORS_PATH",
                str(BACKEND_DIR.parents[1] / "Data Prep" / "games_descriptors.csv"),
            )
        ).expanduser(),
    )


settings = get_settings()
