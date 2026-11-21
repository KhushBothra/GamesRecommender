from __future__ import annotations

import json
import logging
from typing import Any

from Backend.infra.config import settings
from Backend.infra.groq_connector import GroqConnector
from Backend.infra.ollama_connector import OllamaConnector

logger = logging.getLogger(__name__)


class LLMError(RuntimeError):
    """Raised when the selected LLM provider and fallback both fail."""


def chat(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.2,
    max_tokens: int = 500,
) -> str:
    """Use Groq when enabled, with local Ollama as the fallback.

    The caller supplies the system prompt and conversation history. This module
    does not own conversation state or decide when to ask follow-up questions.
    """
    if not messages:
        raise ValueError("messages must contain at least one message")
    if max_tokens < 1:
        raise ValueError("max_tokens must be positive")

    groq_errors: list[str] = []
    if settings.use_groq and settings.groq_api_keys:
        for key_number, api_key in enumerate(settings.groq_api_keys, start=1):
            try:
                connector = GroqConnector(api_key=api_key)
                answer = connector.chat(
                    messages, temperature=temperature, max_tokens=max_tokens
                )
                logger.info(
                    "LLM response generated with Groq (%s; key %d of %d)",
                    connector.model,
                    key_number,
                    len(settings.groq_api_keys),
                )
                return answer
            except Exception as exc:
                groq_errors.append(f"key {key_number}: {exc}")
                logger.warning(
                    "Groq key %d of %d failed; trying the next configured key. Reason: %s",
                    key_number,
                    len(settings.groq_api_keys),
                    exc,
                )

    if not settings.use_groq:
        groq_error = "Groq disabled by USE_GROQ."
    elif not settings.groq_api_keys:
        groq_error = "No GROQ_API_KEY or numbered GROQ_API_KEY<n> values are configured."
    else:
        groq_error = "; ".join(groq_errors)
        logger.warning("All configured Groq keys failed; trying local Ollama.")

    try:
        connector = OllamaConnector()
        answer = connector.chat(messages, temperature=temperature, max_tokens=max_tokens)
        logger.info("LLM response generated with Ollama (%s)", connector.chat_model)
        return answer
    except Exception as ollama_error:
        raise LLMError(
            f"Both configured chat options failed. Groq: {groq_error}; Ollama: {ollama_error}"
        ) from ollama_error


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
