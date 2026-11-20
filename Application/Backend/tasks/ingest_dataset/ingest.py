"""Join the game payload and descriptor CSVs, then upsert them into Qdrant.

Each Qdrant payload includes the display fields plus a ``descriptors`` list of
all active descriptor names. The numeric descriptor weights remain in the
vector and are not duplicated in the payload.

Run from the project root with:

    python -m Backend.tasks.ingest_dataset.ingest

The data paths are resolved relative to the repository root by locating its
``Data Prep`` directory. This works both in the current local Backend folder
and after moving this package under ``Application/Backend``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from Backend.infra.config import settings
from Backend.infra.qdrant_connector import (
    QdrantConnectorError,
    check_qdrant_connection,
    get_qdrant_client,
)


def _find_data_prep_dir() -> Path:
    """Find the project-root Data Prep folder relative to this source file."""
    for parent in Path(__file__).resolve().parents:
        data_prep_dir = parent / "Data Prep"
        if (data_prep_dir / "games_cleaned.csv").is_file() and (
            data_prep_dir / "games_descriptors.csv"
        ).is_file():
            return data_prep_dir
    raise FileNotFoundError(
        "Could not locate the project-root Data Prep folder containing "
        "games_cleaned.csv and games_descriptors.csv."
    )


DATA_PREP_DIR = _find_data_prep_dir()
GAMES_CSV = DATA_PREP_DIR / "games_cleaned.csv"
DESCRIPTORS_CSV = DATA_PREP_DIR / "games_descriptors.csv"
BATCH_SIZE = 256
APP_ID_COLUMN = "AppID"
PAYLOAD_COLUMNS = (
    "AppID",
    "Name",
    "Description",
    "Peak CCU",
    "Popularity Bin",
    "Developers",
    "Publishers",
    "Price",
)


def _read_headers(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        headers = next(csv.reader(stream), None)
    if not headers:
        raise ValueError(f"CSV is empty: {path}")
    return headers


def _normalize_app_id(value: str | None, *, path: Path, row_number: int) -> int:
    if value is None or not value.strip():
        raise ValueError(f"Missing AppID in {path.name}, row {row_number}")
    try:
        app_id = int(value.strip())
    except ValueError as exc:
        raise ValueError(
            f"Invalid AppID {value!r} in {path.name}, row {row_number}"
        ) from exc
    if app_id <= 0:
        raise ValueError(f"AppID must be positive in {path.name}, row {row_number}")
    return app_id


def _load_game_payloads(path: Path) -> dict[int, dict[str, Any]]:
    headers = _read_headers(path)
    missing = [column for column in PAYLOAD_COLUMNS if column not in headers]
    if missing:
        raise ValueError(f"Missing required game columns: {missing}")

    payloads: dict[int, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        for row_number, row in enumerate(reader, start=2):
            app_id = _normalize_app_id(row.get(APP_ID_COLUMN), path=path, row_number=row_number)
            if app_id in payloads:
                raise ValueError(f"Duplicate AppID {app_id} in {path.name}")

            payload = {column: row.get(column) or None for column in PAYLOAD_COLUMNS}
            payload[APP_ID_COLUMN] = app_id
            peak_ccu = payload.get("Peak CCU")
            if peak_ccu is not None:
                try:
                    payload["Peak CCU"] = int(float(peak_ccu))
                except (TypeError, ValueError):
                    pass
            payloads[app_id] = payload
    return payloads


def _descriptor_headers(path: Path) -> list[str]:
    headers = _read_headers(path)
    if headers[0] != APP_ID_COLUMN:
        raise ValueError(f"First descriptor column must be {APP_ID_COLUMN!r}")
    if len(headers) < 2:
        raise ValueError("Descriptor CSV has no feature columns")
    if len(set(headers)) != len(headers):
        raise ValueError("Descriptor CSV contains duplicate column names")
    return headers


def _validate_matching_ids(
    path: Path, expected_ids: set[int], expected_dimensions: int
) -> int:
    seen: set[int] = set()
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        headers = next(reader)
        if len(headers) != expected_dimensions + 1:
            raise ValueError(
                f"Expected {expected_dimensions} descriptors, found {len(headers) - 1}."
            )
        for row_number, row in enumerate(reader, start=2):
            if len(row) != len(headers):
                raise ValueError(
                    f"Unexpected column count in {path.name}, row {row_number}"
                )
            app_id = _normalize_app_id(row[0], path=path, row_number=row_number)
            if app_id in seen:
                raise ValueError(f"Duplicate AppID {app_id} in {path.name}")
            seen.add(app_id)
            if app_id not in expected_ids:
                raise ValueError(f"AppID {app_id} in descriptors has no game payload")
            for value in row[1:]:
                try:
                    number = float(value)
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid descriptor value {value!r} in {path.name}, row {row_number}"
                    ) from exc
                if not math.isfinite(number):
                    raise ValueError(
                        f"Non-finite descriptor value in {path.name}, row {row_number}"
                    )

    missing = expected_ids - seen
    if missing:
        sample = sorted(missing)[:10]
        raise ValueError(
            f"{len(missing)} games have no descriptor row; sample AppIDs: {sample}"
        )
    return len(seen)


def _validate_collection(client: Any, collection_name: str, dimensions: int) -> None:
    if not client.collection_exists(collection_name):
        raise QdrantConnectorError(
            f"Qdrant collection {collection_name!r} does not exist. Create it first."
        )
    info = client.get_collection(collection_name)
    vector_config = info.config.params.vectors
    if isinstance(vector_config, dict):
        if len(vector_config) != 1:
            raise QdrantConnectorError(
                "Expected one unnamed dense vector in the collection."
            )
        vector_config = next(iter(vector_config.values()))
    configured_size = getattr(vector_config, "size", None)
    if configured_size != dimensions:
        raise QdrantConnectorError(
            f"Collection vector size is {configured_size}; dataset has {dimensions} descriptors."
        )


def ingest_dataset(
    games_path: Path = GAMES_CSV,
    descriptors_path: Path = DESCRIPTORS_CSV,
    *,
    batch_size: int = BATCH_SIZE,
) -> int:
    """Validate, join by AppID, and batch-upsert all games into Qdrant."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    payloads = _load_game_payloads(games_path)
    descriptor_headers = _descriptor_headers(descriptors_path)
    dimensions = len(descriptor_headers) - 1
    print(f"Loaded payloads for {len(payloads):,} games.")
    print(f"Descriptor matrix has {dimensions} features.")

    descriptor_count = _validate_matching_ids(
        descriptors_path, set(payloads), dimensions
    )
    print(f"Preflight passed: {descriptor_count:,} matching games, no duplicate IDs.")

    check_qdrant_connection()
    client = get_qdrant_client()
    collection_name = settings.qdrant_collection_name
    _validate_collection(client, collection_name, dimensions)
    print(f"Connected. Upserting into {collection_name!r} in batches of {batch_size}.")

    try:
        from qdrant_client import models
    except ImportError as exc:
        raise QdrantConnectorError(
            "Qdrant support is missing; install the 'qdrant-client' package."
        ) from exc

    batch: list[Any] = []
    uploaded = 0
    with descriptors_path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        next(reader)  # Headers were validated in preflight.
        for row_number, row in enumerate(reader, start=2):
            app_id = int(row[0])
            vector = [float(value) for value in row[1:]]
            active_descriptors = [
                name
                for name, value in zip(descriptor_headers[1:], vector)
                if value > 0
            ]
            payload = dict(payloads[app_id])
            payload["descriptors"] = active_descriptors
            batch.append(
                models.PointStruct(
                    id=app_id,
                    vector=vector,
                    payload=payload,
                )
            )
            if len(batch) >= batch_size:
                client.upsert(collection_name=collection_name, points=batch, wait=True)
                uploaded += len(batch)
                print(f"Uploaded {uploaded:,}/{descriptor_count:,} games.")
                batch.clear()

    if batch:
        client.upsert(collection_name=collection_name, points=batch, wait=True)
        uploaded += len(batch)
        print(f"Uploaded {uploaded:,}/{descriptor_count:,} games.")

    return uploaded


if __name__ == "__main__":
    total = ingest_dataset()
    print(f"Finished: {total:,} games ingested.")
