"""Append completed recommendation runs to a local CSV file."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

logger = logging.getLogger(__name__)

LOG_PATH = Path(__file__).resolve().parents[1] / "logs" / "recommendation_runs.csv"
_LOG_LOCK = Lock()
_FIELDS = (
    "timestamp_utc",
    "opening_user_prompt",
    "generated_query_json",
    "recommended_appids_json",
)


def append_recommendation_run(
    opening_user_prompt: str,
    query: dict[str, Any],
    recommended_appids: Sequence[int],
) -> None:
    """Append one completed run, storing query and ordered AppIDs as JSON cells."""
    if not opening_user_prompt.strip():
        raise ValueError("opening_user_prompt cannot be blank")

    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "opening_user_prompt": opening_user_prompt,
        "generated_query_json": json.dumps(
            query, ensure_ascii=False, separators=(",", ":")
        ),
        "recommended_appids_json": json.dumps(
            [int(appid) for appid in recommended_appids], separators=(",", ":")
        ),
    }

    with _LOG_LOCK:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not LOG_PATH.exists() or LOG_PATH.stat().st_size == 0
        with LOG_PATH.open("a", encoding="utf-8", newline="") as stream:
            # Write a single UTF-8 BOM on first creation so Excel recognizes Unicode.
            if needs_header:
                stream.write("\ufeff")
            writer = csv.DictWriter(stream, fieldnames=_FIELDS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)


__all__ = ["LOG_PATH", "append_recommendation_run"]
