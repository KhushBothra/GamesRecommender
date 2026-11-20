"""Thin FastAPI entry point for the Steam Games Recommender.

Run from the project root with:

    uvicorn Backend.main:app --reload

Conversation, retrieval, and presentation flow is delegated to orchestrator.py.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from Backend.orchestrator import OrchestrationError, orchestrator
from Backend.tasks.conversation.service import (
    ConversationNotFoundError,
    ConversationTaskError,
)

logger = logging.getLogger(__name__)


class UserTextRequest(BaseModel):
    user_text: str = Field(min_length=1, max_length=4000)


class GameRecommendation(BaseModel):
    appid: int
    name: str
    developers: list[str]
    publishers: list[str]
    # Keep price display-ready without assuming currency or numeric formatting.
    price: str | None
    description: str


class ConversationResponse(BaseModel):
    conversation_id: str
    status: Literal["needs_input", "complete"]
    message: str
    recommendations: list[GameRecommendation] = Field(default_factory=list)


app = FastAPI(
    title="Steam Games Recommender API",
    description="Conversational Steam game recommendations.",
    version="1.0.0",
)

# Allow the common local React development server ports.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _run_conversation_action(action, *args) -> dict[str, Any]:
    """Call the orchestrator and translate backend failures to API responses."""
    try:
        return action(*args)
    except ConversationNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (ConversationTaskError, OrchestrationError) as exc:
        logger.exception("Conversation workflow failed")
        raise HTTPException(
            status_code=502,
            detail="The recommendation service could not complete this request.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected conversation workflow error")
        raise HTTPException(
            status_code=502,
            detail="The recommendation service is temporarily unavailable.",
        ) from exc


@app.get("/api/v1/health")
def health() -> dict[str, str]:
    """Return a simple liveness check for the API process."""
    return {"status": "ok", "service": "steam-games-recommender"}


@app.post(
    "/api/v1/conversations",
    response_model=ConversationResponse,
    status_code=200,
)
def start_conversation(request: UserTextRequest) -> ConversationResponse:
    """Start a conversation and return a cross-question or recommendations."""
    return _run_conversation_action(orchestrator.start_conversation, request.user_text)


@app.post(
    "/api/v1/conversations/{conversation_id}/turns",
    response_model=ConversationResponse,
)
def answer_cross_question(
    conversation_id: str,
    request: UserTextRequest,
) -> ConversationResponse:
    """Continue the matching conversation and return its next action."""
    return _run_conversation_action(
        orchestrator.answer_cross_question,
        conversation_id,
        request.user_text,
    )
