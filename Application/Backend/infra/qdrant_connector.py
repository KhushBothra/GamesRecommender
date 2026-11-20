"""Qdrant client adapter. Collection creation waits for the final vector schema."""

from __future__ import annotations

from functools import lru_cache

from Backend.infra.config import settings


class QdrantConnectorError(RuntimeError):
    """Raised when Qdrant cannot be reached or its client is unavailable."""


@lru_cache(maxsize=1)
def get_qdrant_client():
    """Create and cache the configured Qdrant client, without creating a collection."""
    try:
        from qdrant_client import QdrantClient
    except ImportError as exc:
        raise QdrantConnectorError(
            "Qdrant support is missing; install the 'qdrant-client' package."
        ) from exc

    if not settings.qdrant_host:
        raise QdrantConnectorError("QDRANT_HOST is not configured.")
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
        api_key=settings.qdrant_api_key,
    )


def check_qdrant_connection() -> bool:
    """Return whether the configured Qdrant server responds to a read request."""
    try:
        get_qdrant_client().get_collections()
    except Exception as exc:
        raise QdrantConnectorError(f"Could not connect to Qdrant: {exc}") from exc
    return True


def collection_exists() -> bool:
    """Check for the configured collection without creating or modifying it."""
    try:
        return get_qdrant_client().collection_exists(settings.qdrant_collection_name)
    except Exception as exc:
        raise QdrantConnectorError(f"Could not inspect the Qdrant collection: {exc}") from exc
