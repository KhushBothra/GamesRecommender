"""Orchestrate one preference-gathering turn and validate its LLM output."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

from Backend.infra.llm import chat
from Backend.tasks.conversation.prompts import load_prompt, render_prompt
from Backend.tasks.conversation.schemas import (
    LLMDecision,
    ResponseSchemaError,
    parse_decision,
)
from Backend.tasks.conversation.stateful import ConversationSession, conversation_store

logger = logging.getLogger(__name__)
MAX_CLARIFICATION_QUESTIONS = 5
MIN_QUERY_DESCRIPTORS = 10


class ConversationTaskError(RuntimeError):
    """Base error for the conversation task."""


class ConversationNotFoundError(ConversationTaskError):
    """Raised when a conversation does not exist or has expired."""


class ConversationProtocolError(ConversationTaskError):
    """Raised when the LLM cannot produce a valid response after one repair."""


class UnknownDescriptorError(ResponseSchemaError):
    def __init__(self, unknown: list[tuple[str, str]]) -> None:
        self.unknown = unknown
        details = ", ".join(f"{group}: {value!r}" for group, value in unknown)
        super().__init__(f"query contains descriptors outside the allowed vocabulary: {details}")


class InsufficientDescriptorCountError(ResponseSchemaError):
    """Raised when a final query has fewer than ten distinct descriptors."""

    def __init__(self, count: int) -> None:
        self.count = count
        super().__init__(
            f"final query has {count} distinct supported descriptors; "
            f"at least {MIN_QUERY_DESCRIPTORS} are required"
        )


@dataclass(frozen=True)
class ConversationTurn:
    conversation_id: str
    status: Literal["needs_input", "query_ready", "refused"]
    message: str
    query: dict[str, list[str]] | None = None


@lru_cache(maxsize=1)
def _descriptor_vocabularies() -> dict[str, set[str]]:
    """Read the exact three vocabularies embedded in the system prompt."""
    prompt = load_prompt("system.md")
    headings = {
        "tags": "tags",
        "genres": "genres",
        "categories": "categories",
    }
    vocab: dict[str, set[str]] = {}
    for field, heading in headings.items():
        pattern = rf"## Allowed {heading}\s*\([^)]*\)\s*```json\s*(.*?)\s*```"
        match = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            raise ConversationTaskError(
                f"Could not locate the allowed {field} vocabulary in system.md"
            )
        try:
            values = json.loads(match.group(1))
        except json.JSONDecodeError as exc:
            raise ConversationTaskError(
                f"The allowed {field} vocabulary in system.md is invalid JSON"
            ) from exc
        if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
            raise ConversationTaskError(f"The allowed {field} vocabulary must be a string array")
        vocab[field] = set(values)
    return vocab


def _strict_json_object(raw: str) -> dict[str, Any]:
    """Require the entire provider output to be one bare JSON object."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ResponseSchemaError(f"response is not bare JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise ResponseSchemaError("response must be one JSON object")
    return value


def _check_descriptor_membership(decision: LLMDecision) -> None:
    if decision.response_class != 2 or decision.query is None:
        return
    vocab = _descriptor_vocabularies()
    bad: list[tuple[str, str]] = []
    for field in ("tags", "genres", "categories"):
        for value in getattr(decision.query, field):
            if value not in vocab[field]:
                bad.append((field, value))
    if bad:
        raise UnknownDescriptorError(bad)


class ConversationService:
    """Stateful query-generation service, independent of FastAPI routes."""

    def __init__(self) -> None:
        self.store = conversation_store

    def start(self, user_text: str) -> ConversationTurn:
        clean = self._clean_user_text(user_text)
        session = self.store.create()
        session.append("user", clean)
        try:
            return self._advance(session)
        except Exception:
            self.store.remove(session.conversation_id)
            raise

    def answer(self, conversation_id: str, user_text: str) -> ConversationTurn:
        clean = self._clean_user_text(user_text)
        session = self.store.get(conversation_id)
        if session is None:
            raise ConversationNotFoundError("Conversation was not found or has expired.")
        with session.lock:
            session.append("user", clean)
            return self._advance(session)

    def complete(self, conversation_id: str) -> None:
        """Discard conversation context after retrieval and presentation finish."""
        self.store.remove(conversation_id)

    @staticmethod
    def _clean_user_text(user_text: str) -> str:
        if not isinstance(user_text, str) or not user_text.strip():
            raise ValueError("user_text must be a non-empty string")
        clean = user_text.strip()
        if len(clean) > 4000:
            raise ValueError("user_text must be 4000 characters or fewer")
        return clean

    def _advance(self, session: ConversationSession) -> ConversationTurn:
        messages = session.build_llm_messages()
        raw = chat(messages, temperature=0.2, max_tokens=700)
        decision = self._parse_validate_or_repair(raw, messages, session)

        if decision.response_class == 1:
            # The response validator/repair path must not permit a sixth question.
            if session.clarification_questions_asked >= MAX_CLARIFICATION_QUESTIONS:
                raise ConversationProtocolError("LLM attempted to exceed the five-question cap")
            session.clarification_questions_asked += 1
            assert decision.question is not None
            session.append("assistant", decision.question)
            return ConversationTurn(
                conversation_id=session.conversation_id,
                status="needs_input",
                message=decision.question,
            )

        if decision.response_class == 2:
            assert decision.query is not None
            return ConversationTurn(
                conversation_id=session.conversation_id,
                status="query_ready",
                message="I have enough to search for games that fit your preferences.",
                query=decision.query.as_dict(),
            )

        if session.clarification_questions_asked < MAX_CLARIFICATION_QUESTIONS:
            raise ConversationProtocolError(
                "LLM returned the insufficient-descriptor outcome before the clarification limit"
            )
        self.store.remove(session.conversation_id)
        return ConversationTurn(
            conversation_id=session.conversation_id,
            status="refused",
            message=(
                "I couldn't gather enough supported preferences to build a "
                "reliable search. Please start a new search and include your "
                "preferred player mode, connectivity, and game themes or features."
            ),
        )

    def _parse_validate_or_repair(
        self,
        raw: str,
        messages: list[dict[str, str]],
        session: ConversationSession,
    ) -> LLMDecision:
        try:
            return self._validate_payload(_strict_json_object(raw), session)
        except UnknownDescriptorError as exc:
            repair_name = "repair_descriptors.md"
            repair_prompt = render_prompt(
                repair_name,
                PROPOSED_QUERY_JSON=json.dumps(
                    self._try_json_for_repair(raw), ensure_ascii=False
                ),
                INVALID_DESCRIPTORS="\n".join(
                    f"{group}: {value}" for group, value in exc.unknown
                ),
            )
        except InsufficientDescriptorCountError as exc:
            repair_prompt = render_prompt(
                "repair_minimum.md",
                PROPOSED_QUERY_JSON=json.dumps(
                    self._try_json_for_repair(raw), ensure_ascii=False
                ),
                VALIDATION_ERRORS=str(exc),
                QUESTIONS_ASKED=str(session.clarification_questions_asked),
                QUESTIONS_REMAINING=str(
                    MAX_CLARIFICATION_QUESTIONS
                    - session.clarification_questions_asked
                ),
            )
        except (ResponseSchemaError, TypeError, ValueError) as exc:
            repair_prompt = render_prompt(
                "repair_format.md",
                INVALID_OUTPUT=raw,
                VALIDATION_ERRORS=self._format_validation_errors(exc, session),
            )

        # A single correction call prevents repair loops and unbounded token use.
        try:
            repaired_raw = chat(
                [*messages, {"role": "user", "content": repair_prompt}],
                temperature=0,
                max_tokens=700,
            )
            decision = self._validate_payload(_strict_json_object(repaired_raw), session)
        except Exception as exc:
            raise ConversationProtocolError(
                f"LLM response remained invalid after one repair: {exc}"
            ) from exc
        return decision

    def _validate_payload(
        self, payload: dict[str, Any], session: ConversationSession
    ) -> LLMDecision:
        decision = parse_decision(payload)
        _check_descriptor_membership(decision)
        if decision.response_class == 2 and decision.query is not None:
            distinct = {
                value.casefold()
                for field in ("tags", "genres", "categories")
                for value in getattr(decision.query, field)
            }
            if len(distinct) < MIN_QUERY_DESCRIPTORS:
                raise InsufficientDescriptorCountError(len(distinct))
        if decision.response_class == 1 and session.clarification_questions_asked >= MAX_CLARIFICATION_QUESTIONS:
            raise ResponseSchemaError(
                "five clarification questions have already been asked; return a valid query or the insufficient-descriptor outcome"
            )
        if decision.response_class == 4 and session.clarification_questions_asked < MAX_CLARIFICATION_QUESTIONS:
            raise ResponseSchemaError(
                "the insufficient-descriptor outcome is allowed only after five questions have been asked"
            )
        return decision

    @staticmethod
    def _try_json_for_repair(raw: str) -> Any:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {"unparsed_model_output": raw}

    @staticmethod
    def _format_validation_errors(exc: Exception, session: ConversationSession) -> str:
        errors = [str(exc)]
        if session.clarification_questions_asked >= MAX_CLARIFICATION_QUESTIONS:
            errors.append(
                "The five-question limit has been reached. Do not ask another question; "
                "return Class 2 only if both required preference checks are covered and at "
                "least ten distinct supported descriptors map; otherwise return Class 4."
            )
        return "\n".join(errors)


conversation_service = ConversationService()
