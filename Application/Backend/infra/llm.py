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

    use_groq = settings.use_groq
    groq_error = "Groq disabled by USE_GROQ."
    if use_groq:
        try:
            connector = GroqConnector()
            answer = connector.chat(messages, temperature=temperature, max_tokens=max_tokens)
            logger.info("LLM response generated with Groq (%s)", connector.model)
            return answer
        except Exception as exc:
            groq_error = str(exc)
            logger.warning("Groq request failed; trying local Ollama. Reason: %s", groq_error)

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
