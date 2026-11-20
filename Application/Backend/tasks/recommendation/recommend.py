"""Create a weighted descriptor query vector and retrieve its nearest games."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import lru_cache
from typing import Any

from Backend.infra.config import settings
from Backend.infra.embeddings import DescriptorVectorizer
from Backend.infra.qdrant_connector import (
    QdrantConnectorError,
    check_qdrant_connection,
    collection_exists,
    get_qdrant_client,
)

TOP_K = 5
QUERY_FIELDS = {"tags", "genres", "categories"}


class RecommendationError(RuntimeError):
    """Raised when the configured game catalogue cannot be searched."""


@lru_cache(maxsize=1)
def _get_vectorizer() -> DescriptorVectorizer:
    """Load the canonical descriptor-column order once for this process."""
    try:
        return DescriptorVectorizer.from_csv(settings.games_descriptors_path)
    except Exception as exc:
        raise RecommendationError(f"Could not load the descriptor vocabulary: {exc}") from exc


def _query_values(query: Mapping[str, Any]) -> dict[str, list[str]]:
    if not isinstance(query, Mapping):
        raise ValueError("query must be an object with tags, genres, and categories")
    if set(query) != QUERY_FIELDS:
        raise ValueError("query must contain exactly tags, genres, and categories")

    values: dict[str, list[str]] = {}
    for field in ("tags", "genres", "categories"):
        items = query[field]
        if isinstance(items, (str, bytes)) or not isinstance(items, Sequence):
            raise ValueError(f"query.{field} must be an array of strings")
        if any(not isinstance(item, str) or not item.strip() for item in items):
            raise ValueError(f"query.{field} must contain only non-empty strings")
        if any(item != item.strip() for item in items):
            raise ValueError(f"query.{field} values must not have surrounding whitespace")
        values[field] = list(items)

    if not any(values.values()):
        raise ValueError("query must contain at least one descriptor")
    return values


def _qdrant_points(response: Any) -> list[Any]:
    """Extract points from Qdrant's QueryResponse object."""
    points = getattr(response, "points", None)
    if not isinstance(points, list):
        raise RecommendationError("Qdrant returned an unexpected query response")
    return points


def recommend_games(query: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return up to five nearest games for an accepted descriptor query.

    The query must use the conversation task's shape::

        {"tags": [...], "genres": [...], "categories": [...]}

    Tags are assumed to be ordered by user preference; the shared fitted decay
    function assigns their weights. Genres and categories receive weight 1.
    The resulting vector uses the same canonical feature order and 509-D space
    as the ingested games. Qdrant ranks it with the collection's Cosine metric.

    Each result contains its Qdrant ``rank`` and original ``payload``. The raw
    similarity score is intentionally omitted from this task's public result.
    """
    values = _query_values(query)
    vectorizer = _get_vectorizer()
    try:
        query_vector = vectorizer.encode_query(
            tags=values["tags"],
            genres=values["genres"],
            categories=values["categories"],
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Could not encode descriptor query: {exc}") from exc

    if len(query_vector) != vectorizer.dimension:
        raise RecommendationError(
            f"Query vector has {len(query_vector)} dimensions; expected {vectorizer.dimension}"
        )

    try:
        check_qdrant_connection()
        if not collection_exists():
            raise RecommendationError(
                f"Qdrant collection {settings.qdrant_collection_name!r} does not exist"
            )
        response = get_qdrant_client().query_points(
            collection_name=settings.qdrant_collection_name,
            query=query_vector,
            limit=TOP_K,
            with_payload=True,
            with_vectors=False,
        )
    except (QdrantConnectorError, RecommendationError):
        raise
    except Exception as exc:
        raise RecommendationError(f"Qdrant recommendation search failed: {exc}") from exc

    results: list[dict[str, Any]] = []
    for rank, point in enumerate(_qdrant_points(response), start=1):
        payload = getattr(point, "payload", None)
        if not isinstance(payload, dict):
            raise RecommendationError(f"Qdrant result at rank {rank} has no payload")
        if payload.get("AppID", payload.get("appid")) is None:
            raise RecommendationError(f"Qdrant result at rank {rank} has no AppID")
        results.append(
            {
                "rank": rank,
                "payload": dict(payload),
            }
        )
    return results
