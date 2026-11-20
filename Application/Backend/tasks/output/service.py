"""Generate and validate concise explanations for retrieved game results."""

from __future__ import annotations

import json
import logging
from typing import Any

from Backend.infra.llm import chat
from Backend.tasks.output.prompts import load_prompt, render_prompt
from Backend.tasks.output.schemas import GameExplanation, OutputSchemaError, parse_output

logger = logging.getLogger(__name__)

MAX_INPUT_GAMES = 30
MAX_EXPLANATION_TOKENS_PER_GAME = 120
BASE_OUTPUT_TOKENS = 300
MAX_OUTPUT_TOKENS = 5000


class OutputTaskError(RuntimeError):
    """Base error for result-presentation failures."""


class OutputProtocolError(OutputTaskError):
    """Raised when the LLM output remains invalid after one repair attempt."""


def _normalize_appid(value: Any, index: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"games[{index}] has an invalid AppID")
    try:
        appid = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"games[{index}] has an invalid AppID") from exc
    if appid <= 0 or str(value).strip() not in {str(appid), f"{appid}.0"}:
        raise ValueError(f"games[{index}] has an invalid AppID")
    return appid


def _validate_inputs(
    conversation: list[dict[str, str]],
    final_query: dict[str, Any],
    games: list[dict[str, Any]],
) -> list[int]:
    if not isinstance(conversation, list):
        raise ValueError("conversation must be a role-based message list")
    for index, message in enumerate(conversation):
        if (
            not isinstance(message, dict)
            or set(message) != {"role", "content"}
            or not isinstance(message["role"], str)
            or message["role"] not in {"user", "assistant"}
            or not isinstance(message["content"], str)
        ):
            raise ValueError(f"conversation[{index}] must have user/assistant role and string content")

    if not isinstance(final_query, dict):
        raise ValueError("final_query must be a JSON object")
    if not isinstance(games, list):
        raise ValueError("games must be an ordered list")
    if len(games) > MAX_INPUT_GAMES:
        raise ValueError(f"at most {MAX_INPUT_GAMES} games can be explained in one request")

    appids: list[int] = []
    for index, game in enumerate(games):
        if not isinstance(game, dict) or not isinstance(game.get("payload"), dict):
            raise ValueError(f"games[{index}] must contain a Qdrant payload object")
        payload = game["payload"]
        appid = _normalize_appid(payload.get("AppID", payload.get("appid")), index)
        if appid in appids:
            raise ValueError(f"duplicate AppID {appid} in retrieved games")
        appids.append(appid)
        try:
            json.dumps(game, ensure_ascii=False, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"games[{index}] is not JSON-serializable") from exc

    try:
        json.dumps(final_query, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("final_query is not JSON-serializable") from exc
    return appids


def _strict_json_object(raw: str) -> dict[str, Any]:
    try:
        result = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OutputSchemaError(f"response is not bare JSON: {exc.msg}") from exc
    if not isinstance(result, dict):
        raise OutputSchemaError("response must be one JSON object")
    return result


def _build_user_input(
    conversation: list[dict[str, str]],
    final_query: dict[str, Any],
    games: list[dict[str, Any]],
) -> str:
    data = {
        "conversation": conversation,
        "final_query": final_query,
        "games": games,
    }
    return json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def _parse_or_repair(
    raw: str,
    *,
    messages: list[dict[str, str]],
    expected_appids: list[int],
) -> list[GameExplanation]:
    try:
        return parse_output(_strict_json_object(raw), expected_appids)
    except (OutputSchemaError, TypeError, ValueError) as exc:
        repair_message = render_prompt(
            "repair_format.md",
            INVALID_OUTPUT=raw,
            VALIDATION_ERRORS=str(exc),
        )

    try:
        repaired = chat(
            [*messages, {"role": "user", "content": repair_message}],
            temperature=0,
            max_tokens=min(
                MAX_OUTPUT_TOKENS,
                BASE_OUTPUT_TOKENS + len(expected_appids) * MAX_EXPLANATION_TOKENS_PER_GAME,
            ),
        )
        return parse_output(_strict_json_object(repaired), expected_appids)
    except Exception as exc:
        raise OutputProtocolError(
            f"LLM output remained invalid after one repair attempt: {exc}"
        ) from exc


class OutputService:
    """Create grounded explanations and attach them to original game records."""

    def generate(
        self,
        conversation: list[dict[str, str]],
        final_query: dict[str, Any],
        games: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Return input games in original order, with a generated explanation added.

        Each input game has ``rank`` and ``payload`` keys, matching the output
        prompt's Qdrant result contract. The LLM returns only AppIDs and
        explanations; original payload fields remain owned by the backend.
        """
        expected_appids = _validate_inputs(conversation, final_query, games)
        if not games:
            return []

        messages = [
            {"role": "system", "content": load_prompt("system.md")},
            {
                "role": "user",
                "content": _build_user_input(conversation, final_query, games),
            },
        ]
        max_tokens = min(
            MAX_OUTPUT_TOKENS,
            BASE_OUTPUT_TOKENS + len(games) * MAX_EXPLANATION_TOKENS_PER_GAME,
        )
        raw = chat(messages, temperature=0, max_tokens=max_tokens)
        explanations = _parse_or_repair(
            raw,
            messages=messages,
            expected_appids=expected_appids,
        )

        output: list[dict[str, Any]] = []
        for game, explanation in zip(games, explanations):
            # Copy both wrapper and payload to avoid mutating Qdrant's result.
            result = dict(game)
            result["payload"] = dict(game["payload"])
            result["explanation"] = explanation.explanation
            output.append(result)

        logger.info("Generated explanations for %d retrieved games", len(output))
        return output


output_service = OutputService()
