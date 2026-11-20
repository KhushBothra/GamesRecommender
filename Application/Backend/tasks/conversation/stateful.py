"""Temporary conversation history and system-prompt message assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import RLock
from time import monotonic
from typing import Literal
from uuid import uuid4

from Backend.tasks.conversation.prompts import load_prompt

MessageRole = Literal["user", "assistant"]
SESSION_TTL_SECONDS = 30 * 60


@dataclass
class ConversationSession:
    conversation_id: str = field(default_factory=lambda: uuid4().hex)
    messages: list[dict[str, str]] = field(default_factory=list)
    clarification_questions_asked: int = 0
    last_activity: float = field(default_factory=monotonic)
    lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    def append(self, role: MessageRole, content: str) -> None:
        clean = content.strip()
        if not clean:
            raise ValueError("conversation message cannot be blank")
        self.messages.append({"role": role, "content": clean})
        self.last_activity = monotonic()

    def build_llm_messages(self) -> list[dict[str, str]]:
        """Return system prompt followed by the complete role-based transcript."""
        return [
            {"role": "system", "content": load_prompt("system.md")},
            *[dict(message) for message in self.messages],
        ]


class ConversationStore:
    """Thread-safe in-memory sessions with idle expiry and completion cleanup."""

    def __init__(self, ttl_seconds: int = SESSION_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def _remove_expired(self, now: float) -> None:
        expired = [
            session_id
            for session_id, session in self._sessions.items()
            if now - session.last_activity >= self._ttl_seconds
        ]
        for session_id in expired:
            self._sessions.pop(session_id, None)

    def create(self) -> ConversationSession:
        with self._lock:
            self._remove_expired(monotonic())
            session = ConversationSession()
            self._sessions[session.conversation_id] = session
            return session

    def get(self, conversation_id: str) -> ConversationSession | None:
        with self._lock:
            self._remove_expired(monotonic())
            return self._sessions.get(conversation_id)

    def remove(self, conversation_id: str) -> None:
        with self._lock:
            self._sessions.pop(conversation_id, None)

    def cleanup_expired(self) -> int:
        with self._lock:
            before = len(self._sessions)
            self._remove_expired(monotonic())
            return before - len(self._sessions)


conversation_store = ConversationStore()
