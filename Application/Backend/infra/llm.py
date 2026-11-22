from __future__ import annotations

import json
import logging
from typing import Any

from Backend.infra.config import settings
from Backend.infra.groq_connector import GroqConnector

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when Groq is unconfigured or all configured keys fail."""


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 500,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Use Groq, trying each configured API key in order.

    The caller supplies the system prompt, conversation history, and optional
    response schema. This module
    does not own conversation state or decide when to ask follow-up questions.
    """
    if not messages:
        raise ValueError("messages must contain at least one message")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    if not settings.groq_api_keys:
        raise LLMError("No GROQ_API_KEY or numbered GROQ_API_KEY<n> values are configured.")

    groq_errors: list[str] = []
    options: dict[str, Any] = {"temperature": temperature, "max_tokens": max_tokens}
    if response_schema is not None:
        options["response_schema"] = response_schema
    last_error: Exception | None = None
    for key_number, api_key in enumerate(settings.groq_api_keys, start=1):
        try:
            connector = GroqConnector(api_key=api_key)
            answer = connector.chat(messages, **options)
            logger.info(
                "LLM response generated with Groq (%s; key %d of %d)",
                connector.model,
                key_number,
                len(settings.groq_api_keys),
            )
            return answer
        except Exception as exc:
            last_error = exc
            groq_errors.append(f"key {key_number}: {exc}")
            logger.warning(
                "Groq key %d of %d failed. Reason: %s",
                key_number,
                len(settings.groq_api_keys),
                exc,
            )

    raise LLMError(
        "Groq chat failed for all configured API keys. " + "; ".join(groq_errors)
    ) from last_error


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, accepting a single surrounding Markdown fence."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[-1]
        if candidate.endswith("```"):
            candidate = candidate[:-3].strip()
    parsed = json.loads(candidate)
    if not isinstance(parsed, dict):
        raise ValueError("Expected the LLM response to be a JSON object.")
    return parsed
