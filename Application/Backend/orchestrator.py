"""Coordinate conversation, recommendation retrieval, and result presentation.

The FastAPI entry point should call this module for conversation requests. It
owns workflow decisions while individual task modules own their specific work.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass
from typing import Any, Literal

from Backend.infra.steam_artwork import steam_artwork

from Backend.tasks.conversation.service import (
    ConversationService,
    ConversationTurn,
    conversation_service,
)
from Backend.tasks.output.service import OutputService, output_service
from Backend.tasks.recommendation.recommend import recommend_games

logger = logging.getLogger(__name__)

ConversationStatus = Literal["needs_input", "complete"]


@dataclass(frozen=True)
class OrchestratorResponse:
    """Response shape expected by the current frontend API contract."""

    conversation_id: str
    status: ConversationStatus
    message: str
    recommendations: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "status": self.status,
            "message": self.message,
            "recommendations": self.recommendations,
        }


class OrchestrationError(RuntimeError):
    """Raised when a terminal recommendation flow cannot be completed."""


def _display_list(value: Any) -> list[str]:
    """Convert a CSV/Qdrant developer or publisher value to display strings.

    JSON/Python-list-shaped CSV values are parsed as lists. Other strings are
    kept whole so commas inside a company name are not mistaken for separators.
    """
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if not isinstance(value, str):
        value = str(value)
    clean = value.strip()
    if not clean:
        return []

    if clean.startswith(("[", "(")) and clean.endswith(("]", ")")):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(clean)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
    return [clean]


def _positive_appid(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("game payload contains an invalid AppID")
    try:
        appid = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("game payload contains an invalid AppID") from exc
    if appid <= 0:
        raise ValueError("game payload contains an invalid AppID")
    return appid


def _frontend_game(result: dict[str, Any]) -> dict[str, Any]:
    """Map one preserved Qdrant payload plus explanation to the API game shape."""
    payload = result.get("payload")
    if not isinstance(payload, dict):
        raise OrchestrationError("recommendation result is missing its game payload")

    name = payload.get("Name")
    if not isinstance(name, str) or not name.strip():
        raise OrchestrationError("recommended game payload is missing its Name")

    explanation = result.get("explanation")
    if explanation is not None and not isinstance(explanation, str):
        raise OrchestrationError("generated game explanation must be a string or null")

    price = payload.get("Price")
    if price is not None:
        price = str(price)

    return {
        "appid": _positive_appid(payload.get("AppID", payload.get("appid"))),
        "name": name.strip(),
        "developers": _display_list(payload.get("Developers")),
        "publishers": _display_list(payload.get("Publishers")),
        "price": price,
        # The LLM explanation is the display description in the current API.
        "description": explanation or "",
    }


class ApplicationOrchestrator:
    """Run the complete request flow behind the FastAPI route layer."""

    def __init__(
        self,
        conversations: ConversationService = conversation_service,
        presenter: OutputService = output_service,
    ) -> None:
        self.conversations = conversations
        self.presenter = presenter

    def start_conversation(self, user_text: str) -> dict[str, Any]:
        """Begin a conversation, returning a question or final recommendations."""
        turn = self.conversations.start(user_text)
        return self._handle_turn(turn)

    def answer_cross_question(
        self, conversation_id: str, user_text: str
    ) -> dict[str, Any]:
        """Continue a conversation and return its next question or results."""
        turn = self.conversations.answer(conversation_id, user_text)
        return self._handle_turn(turn)

    def _handle_turn(self, turn: ConversationTurn) -> dict[str, Any]:
        if turn.status == "needs_input":
            return OrchestratorResponse(
                conversation_id=turn.conversation_id,
                status="needs_input",
                message=turn.message,
                recommendations=[],
            ).as_dict()

        if turn.status == "refused":
            return OrchestratorResponse(
                conversation_id=turn.conversation_id,
                status="complete",
                message=turn.message,
                recommendations=[],
            ).as_dict()

        if turn.status != "query_ready" or turn.query is None:
            raise OrchestrationError("conversation task returned an unsupported state")

        session = self.conversations.store.get(turn.conversation_id)
        if session is None:
            raise OrchestrationError("conversation context expired before retrieval")

        # Snapshot the transcript under the session lock. The accepted query is
        # supplied separately because the conversation task returns it as data.
        with session.lock:
            transcript = [dict(message) for message in session.messages]

        try:
            retrieved = recommend_games(turn.query)
            presented = self.presenter.generate(transcript, turn.query, retrieved)
            recommendations = [_frontend_game(game) for game in presented]
        except Exception as exc:
            # A final query is terminal. Do not leave an unusable session around
            # if retrieval or presentation fails.
            self.conversations.complete(turn.conversation_id)
            logger.exception("Could not complete recommendation flow")
            raise OrchestrationError(f"Could not generate recommendations: {exc}") from exc

        self.conversations.complete(turn.conversation_id)
        try:
            recommendations = steam_artwork.enrich(recommendations)
        except Exception:
            logger.warning("Could not enrich artwork; returning text cards", exc_info=True)
            recommendations = [{**game, "image_url": None} for game in recommendations]
        if recommendations:
            message = f"Here are {len(recommendations)} games that may fit your preferences."
        else:
            message = "I couldn't find any games for that query. Try adjusting your preferences."
        return OrchestratorResponse(
            conversation_id=turn.conversation_id,
            status="complete",
            message=message,
            recommendations=recommendations,
        ).as_dict()


orchestrator = ApplicationOrchestrator()

__all__ = [
    "ApplicationOrchestrator",
    "OrchestrationError",
    "OrchestratorResponse",
    "orchestrator",
]
