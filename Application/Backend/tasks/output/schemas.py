"""Validation for the result-presentation LLM response."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class OutputSchemaError(ValueError):
    """Raised when generated output does not match the required JSON contract."""


@dataclass(frozen=True)
class GameExplanation:
    appid: int
    explanation: str | None


def parse_output(data: Any, expected_appids: list[int]) -> list[GameExplanation]:
    """Validate exact game count, order, IDs, and allowed response fields."""
    if not isinstance(data, dict) or set(data) != {"games"}:
        raise OutputSchemaError("response must contain exactly one top-level 'games' field")

    games = data["games"]
    if not isinstance(games, list):
        raise OutputSchemaError("games must be an array")
    if len(games) != len(expected_appids):
        raise OutputSchemaError(
            f"expected {len(expected_appids)} game outputs, received {len(games)}"
        )

    parsed: list[GameExplanation] = []
    for index, (item, expected_appid) in enumerate(zip(games, expected_appids)):
        if not isinstance(item, dict) or set(item) != {"appid", "explanation"}:
            raise OutputSchemaError(
                f"games[{index}] must contain exactly 'appid' and 'explanation'"
            )

        appid = item["appid"]
        if type(appid) is not int or appid != expected_appid:
            raise OutputSchemaError(
                f"games[{index}].appid must be {expected_appid} and remain in input order"
            )

        explanation = item["explanation"]
        if explanation is not None and (
            not isinstance(explanation, str) or not explanation.strip()
        ):
            raise OutputSchemaError(
                f"games[{index}].explanation must be a non-empty string or null"
            )
        parsed.append(
            GameExplanation(
                appid=appid,
                explanation=explanation.strip() if isinstance(explanation, str) else None,
            )
        )

    return parsed
