"""Validation for the conversation LLM's JSON response contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any

DECISION_JSON_SCHEMA: dict[str, Any] = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "class": {"type": "integer", "enum": [1]},
                "question": {"type": "string"},
            },
            "required": ["class", "question"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "class": {"type": "integer", "enum": [2]},
                "query": {
                    "type": "object",
                    "properties": {
                        group: {"type": "array", "items": {"type": "string"}}
                        for group in ("tags", "genres", "categories")
                    },
                    "required": ["tags", "genres", "categories"],
                    "additionalProperties": False,
                },
            },
            "required": ["class", "query"],
            "additionalProperties": False,
        },
        {
            "type": "object",
            "properties": {
                "class": {"type": "integer", "enum": [4]},
                "reason": {"type": "string", "enum": ["INSUFFICIENT_QUERY_CONTEXT"]},
            },
            "required": ["class", "reason"],
            "additionalProperties": False,
        },
    ]
}


def decision_json_schema(*, allow_clarification: bool) -> dict[str, Any]:
    """Constrain generation to outcomes legal at this conversation turn."""
    allowed_classes = {1, 2} if allow_clarification else {2, 4}
    return {
        "anyOf": [
            deepcopy(branch)
            for branch in DECISION_JSON_SCHEMA["anyOf"]
            if branch["properties"]["class"]["enum"][0] in allowed_classes
        ]
    }


class ResponseSchemaError(ValueError):
    """Raised when a model response does not match the expected JSON shape."""


@dataclass(frozen=True)
class DescriptorQuery:
    tags: list[str]
    genres: list[str]
    categories: list[str]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "tags": list(self.tags),
            "genres": list(self.genres),
            "categories": list(self.categories),
        }


@dataclass(frozen=True)
class LLMDecision:
    response_class: int
    question: str | None = None
    query: DescriptorQuery | None = None
    reason: str | None = None


def _string_array(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ResponseSchemaError(f"{field} must be an array of strings")
    if any(not item.strip() for item in value):
        raise ResponseSchemaError(f"{field} cannot contain blank strings")
    if any(item != item.strip() for item in value):
        raise ResponseSchemaError(f"{field} values must not have surrounding whitespace")
    if len({item.casefold() for item in value}) != len(value):
        raise ResponseSchemaError(f"{field} cannot contain duplicate descriptors")
    return list(value)


def parse_decision(data: Any) -> LLMDecision:
    """Validate the strict class 1/2/4 JSON protocol from system.md."""
    if not isinstance(data, dict):
        raise ResponseSchemaError("response must be a JSON object")

    response_class = data.get("class")
    if type(response_class) is not int or response_class not in {1, 2, 4}:
        raise ResponseSchemaError("class must be integer 1, 2, or 4")

    if response_class == 1:
        if set(data) != {"class", "question"}:
            raise ResponseSchemaError("class 1 must contain exactly class and question")
        question = data.get("question")
        if not isinstance(question, str) or not question.strip():
            raise ResponseSchemaError("question must be a non-empty string")
        return LLMDecision(response_class=1, question=question.strip())

    if response_class == 2:
        if set(data) != {"class", "query"}:
            raise ResponseSchemaError("class 2 must contain exactly class and query")
        query_data = data.get("query")
        if not isinstance(query_data, dict):
            raise ResponseSchemaError("query must be an object")
        required = {"tags", "genres", "categories"}
        if set(query_data) != required:
            raise ResponseSchemaError(
                "query must contain exactly tags, genres, and categories arrays"
            )
        query = DescriptorQuery(
            tags=_string_array(query_data["tags"], "query.tags"),
            genres=_string_array(query_data["genres"], "query.genres"),
            categories=_string_array(query_data["categories"], "query.categories"),
        )
        if not (query.tags or query.genres or query.categories):
            raise ResponseSchemaError("query must contain at least one descriptor")
        return LLMDecision(response_class=2, query=query)

    if set(data) != {"class", "reason"}:
        raise ResponseSchemaError("class 4 must contain exactly class and reason")
    if data.get("reason") != "INSUFFICIENT_QUERY_CONTEXT":
        raise ResponseSchemaError(
            "class 4 reason must be INSUFFICIENT_QUERY_CONTEXT"
        )
    return LLMDecision(response_class=4, reason="INSUFFICIENT_QUERY_CONTEXT")
