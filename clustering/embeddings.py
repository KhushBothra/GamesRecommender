"""Validated, reusable description embeddings for the clustering pipeline.

Importing this module does not import torch or sentence-transformers.  A
compatible precomputed matrix or cache can therefore be used without either.
"""

from __future__ import annotations

import hashlib
from html.parser import HTMLParser
from importlib import metadata as package_metadata
import json
import os
from pathlib import Path
import platform
import re
from typing import Any, Iterator
import uuid

import numpy as np


SCHEMA_VERSION = 1
CLEANING_VERSION = "html_text_without_script_style_whitespace_v1"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.hidden: list[str] = []

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in {"script", "style"}:
            self.hidden.append(tag)
        elif not self.hidden:
            self.parts.append(" ")

    def handle_startendtag(self, tag: str, attrs: Any) -> None:
        if not self.hidden:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if self.hidden:
            if tag == self.hidden[-1]:
                self.hidden.pop()
        else:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        if not self.hidden:
            self.parts.append(data)


def clean_description(value: Any) -> str:
    """Remove HTML, script/style content and repeated whitespace."""
    if not isinstance(value, str):
        raise ValueError("Descriptions must be non-missing strings.")
    parser = _TextExtractor()
    parser.feed(value)
    parser.close()
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _prepared_input(frame: Any) -> tuple[np.ndarray, list[str], str, str]:
    try:
        raw_ids = list(frame["AppID"])
        raw_texts = list(frame["Description"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Input must contain AppID and Description columns.") from exc
    if not raw_ids or len(raw_ids) != len(raw_texts):
        raise ValueError("Description input must contain matching, non-empty rows.")
    ids: list[int] = []
    texts: list[str] = []
    text_digest = hashlib.sha256()
    for position, (raw_id, raw_text) in enumerate(zip(raw_ids, raw_texts)):
        try:
            appid = int(raw_id)
        except (ValueError, TypeError, OverflowError) as exc:
            raise ValueError(f"Invalid AppID at row {position}: {raw_id!r}.") from exc
        if isinstance(raw_id, (bool, np.bool_)) or (
            isinstance(raw_id, (float, np.floating)) and raw_id != appid
        ) or appid <= 0:
            raise ValueError(f"Invalid AppID at row {position}: {raw_id!r}.")
        try:
            text = clean_description(raw_text)
        except ValueError as exc:
            raise ValueError(f"Missing description for AppID {appid}.") from exc
        if not text:
            raise ValueError(f"Empty cleaned description for AppID {appid}.")
        encoded = text.encode("utf-8")
        text_digest.update(len(encoded).to_bytes(8, byteorder="little", signed=False))
        text_digest.update(encoded)
        ids.append(appid)
        texts.append(text)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate AppIDs in description input.")
    try:
        id_array = np.asarray(ids, dtype="<i8")
    except (ValueError, OverflowError) as exc:
        raise ValueError("AppIDs must fit signed 64-bit integers.") from exc
    return id_array, texts, hashlib.sha256(id_array.tobytes()).hexdigest(), text_digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read embedding metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Embedding metadata must be a JSON object: {path}.")
    return data


def _close_memmap(array: Any) -> None:
    mapped_file = getattr(array, "_mmap", None)
    if mapped_file is not None:
        mapped_file.close()


def _load_verified(
    path: Path,
    expected: dict[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    metadata_path = path.with_suffix(".json")
    if not path.is_file() or not metadata_path.is_file():
        raise ValueError(f"Embedding matrix and metadata must both exist: {path} and {metadata_path}.")
    metadata = _read_json(metadata_path)
    for name, value in expected.items():
        if name not in metadata or metadata[name] != value:
            raise ValueError(
                f"Embedding provenance mismatch for {name} in {metadata_path}: "
                f"expected {value!r}, got {metadata.get(name)!r}."
            )
    if metadata.get("normalized") is not True:
        raise ValueError(f"Embeddings must declare normalized=true in {metadata_path}.")
    if "cleaning_version" in metadata and metadata["cleaning_version"] != CLEANING_VERSION:
        raise ValueError(f"Description cleaning version mismatch in {metadata_path}.")
    dimension = metadata.get("dimension")
    if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 1:
        raise ValueError(f"Invalid embedding dimension in {metadata_path}.")
    checksum = metadata.get("embedding_sha256")
    if not isinstance(checksum, str) or not re.fullmatch(r"[0-9a-f]{64}", checksum):
        raise ValueError(f"Missing or invalid embedding_sha256 in {metadata_path}.")
    if _sha256(path) != checksum:
        raise ValueError(f"Embedding file checksum mismatch: {path}.")
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot load embedding matrix {path}: {exc}") from exc
    try:
        if not isinstance(array, np.ndarray) or array.shape != (expected["rows"], dimension):
            raise ValueError(f"Embedding shape must be ({expected['rows']}, {dimension}): {path}.")
        if not np.issubdtype(array.dtype, np.floating):
            raise ValueError(f"Embedding matrix must have a floating-point dtype: {path}.")
        for start in range(0, len(array), 8192):
            block = array[start : start + 8192]
            if not np.isfinite(block).all():
                raise ValueError(f"Embedding matrix contains non-finite values: {path}.")
            norms = np.linalg.norm(np.asarray(block, dtype=np.float64), axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-4, atol=1e-5):
                raise ValueError(f"Embedding rows must have unit L2 norm: {path}.")
        if array.dtype != np.float32:
            converted = np.asarray(array, dtype=np.float32)
            _close_memmap(array)
            array = converted
        return array, metadata
    except Exception:
        _close_memmap(array)
        raise


def _normalize(batch: Any, rows: int) -> np.ndarray:
    result = np.asarray(batch, dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != rows or result.shape[1] < 1:
        raise ValueError(f"Encoder returned an invalid matrix shape: {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("Encoder returned non-finite values.")
    norms = np.linalg.norm(np.asarray(result, dtype=np.float64), axis=1)
    if not np.isfinite(norms).all() or (norms <= 0).any():
        raise ValueError("Encoder returned an empty or invalid description vector.")
    return np.asarray(result / norms[:, None], dtype=np.float32)


def _encode(model: Any, texts: list[str], batch_size: int) -> np.ndarray:
    result = np.asarray(model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=False,
    ), dtype=np.float32)
    if result.ndim != 2 or result.shape[0] != len(texts) or result.shape[1] < 1:
        raise ValueError(f"Encoder returned an invalid matrix shape: {result.shape}.")
    if not np.isfinite(result).all():
        raise ValueError("Encoder returned non-finite values.")
    return result


def _chunked_batch(model: Any, texts: list[str], batch_size: int, max_seq_length: int) -> np.ndarray:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None:
        raise ValueError("Chunked descriptions require the encoder's tokenizer.")
    token_limit = max_seq_length - tokenizer.num_special_tokens_to_add(pair=False)
    if token_limit < 1:
        raise ValueError("max_seq_length leaves no room for description tokens.")
    chunks: list[str] = []
    owners: list[int] = []
    for owner, text in enumerate(texts):
        tokens = tokenizer.encode(text, add_special_tokens=False, truncation=False)
        if not tokens:
            raise ValueError("A cleaned description produced no tokens.")
        for start in range(0, len(tokens), token_limit):
            chunk = tokenizer.decode(
                tokens[start : start + token_limit],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if not chunk.strip():
                raise ValueError("Tokenizer produced an empty description chunk.")
            chunks.append(chunk)
            owners.append(owner)
    sums: np.ndarray | None = None
    counts = np.bincount(owners, minlength=len(texts)).astype(np.float64)
    for start in range(0, len(chunks), batch_size):
        encoded = _encode(model, chunks[start : start + batch_size], batch_size)
        if sums is None:
            sums = np.zeros((len(texts), encoded.shape[1]), dtype=np.float64)
        elif encoded.shape[1] != sums.shape[1]:
            raise ValueError("Encoder dimension changed between chunks.")
        np.add.at(sums, owners[start : start + len(encoded)], encoded)
    if sums is None or (counts <= 0).any():
        raise ValueError("No valid description chunks were encoded.")
    return _normalize(sums / counts[:, None], len(texts))


def _batches(
    model: Any,
    texts: list[str],
    batch_size: int,
    max_seq_length: int,
    chunked: bool,
) -> Iterator[tuple[int, np.ndarray]]:
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        block = texts[start : start + batch_size]
        if chunked:
            encoded = _chunked_batch(model, block, batch_size, max_seq_length)
        else:
            encoded = _normalize(_encode(model, block, batch_size), len(block))
        yield start, encoded
        if batch_number == 1 or batch_number % 10 == 0 or start + len(block) == len(texts):
            print(f"[descriptions] encoded {start + len(block):,}/{len(texts):,} games", flush=True)


def _versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": platform.python_version()}
    for name in ("numpy", "sentence-transformers", "transformers", "torch"):
        try:
            result[name] = package_metadata.version(name)
        except package_metadata.PackageNotFoundError:
            result[name] = None
    return result


def _resolved_revision(model: Any) -> str | None:
    try:
        value = model[0].auto_model.config._commit_hash
    except (AttributeError, IndexError, TypeError, KeyError):
        return None
    return str(value) if value else None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(f"{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as target:
            json.dump(data, target, ensure_ascii=False, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_descriptions(frame: Any, args: Any) -> tuple[np.ndarray, dict[str, Any]]:
    """Return verified normalized float32 rows in the input frame's AppID order.

    External matrices need a same-stem JSON sidecar.  Cache compatibility is
    based on cleaned text and AppID order, not just array dimensions.  ``force``
    permits replacing an incompatible/incomplete cache; valid caches are reused.
    """
    _, texts, app_ids_sha256, description_text_sha256 = _prepared_input(frame)
    max_seq_length = getattr(args, "max_seq_length", 256)
    batch_size = getattr(args, "batch_size", 64)
    if isinstance(max_seq_length, bool) or not isinstance(max_seq_length, int) or max_seq_length < 1:
        raise ValueError("max_seq_length must be a positive integer.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer.")
    expected = {
        "schema_version": SCHEMA_VERSION,
        "app_ids_sha256": app_ids_sha256,
        "description_text_sha256": description_text_sha256,
        "model_id": str(getattr(args, "description_model", DEFAULT_MODEL)),
        "model_revision": getattr(args, "model_revision", None),
        "max_seq_length": max_seq_length,
        "pooling": "mean_chunks" if getattr(args, "chunk_descriptions", False) else "truncate",
        "rows": len(texts),
    }
    external = getattr(args, "description_embeddings", None)
    if external is not None:
        path = Path(external).expanduser().resolve()
        if path.suffix.lower() != ".npy":
            raise ValueError("Precomputed description embeddings must be a .npy file.")
        return _load_verified(path, expected)

    identity = {**expected, "cleaning_version": CLEANING_VERSION, "normalized": True}
    fingerprint = hashlib.sha256(json.dumps(
        identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")).hexdigest()
    cache_dir = Path(args.cache_dir).expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"descriptions_{fingerprint}.npy"
    metadata_path = path.with_suffix(".json")
    if path.exists() or metadata_path.exists():
        try:
            return _load_verified(path, expected)
        except ValueError as exc:
            if not getattr(args, "force", False):
                raise ValueError(f"{exc} Use a fresh cache directory or --force to rebuild it.") from exc

    try:
        from sentence_transformers import SentenceTransformer
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Description encoding requires sentence-transformers. Install "
            "clustering/requirements.txt or supply verified precomputed embeddings."
        ) from exc
    threads = getattr(args, "threads", 8)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ValueError("threads must be a positive integer.")
    torch.set_num_threads(threads)
    try:
        model = SentenceTransformer(
            expected["model_id"],
            revision=expected["model_revision"],
            device=getattr(args, "device", "cpu"),
            cache_folder=str(cache_dir / "huggingface"),
            local_files_only=bool(getattr(args, "offline", False)),
        )
    except Exception as exc:
        offline_note = " Offline mode requires a locally available model." if getattr(args, "offline", False) else ""
        raise RuntimeError(f"Could not load description model {expected['model_id']!r}.{offline_note} {exc}") from exc
    model.max_seq_length = max_seq_length
    partial = path.with_name(f"{path.stem}.{uuid.uuid4().hex}.partial.npy")
    storage: np.memmap | None = None
    dimension: int | None = None
    try:
        for start, encoded in _batches(
            model, texts, batch_size, max_seq_length, expected["pooling"] == "mean_chunks",
        ):
            if storage is None:
                dimension = encoded.shape[1]
                storage = np.lib.format.open_memmap(
                    partial, mode="w+", dtype=np.float32, shape=(len(texts), dimension),
                )
            elif encoded.shape[1] != dimension:
                raise ValueError("Encoder dimension changed between batches.")
            storage[start : start + len(encoded)] = encoded
        if storage is None or dimension is None:
            raise ValueError("No description embeddings were produced.")
        storage.flush()
        _close_memmap(storage)
        storage = None
        metadata = {
            **identity,
            "dimension": dimension,
            "identity_fingerprint": fingerprint,
            "embedding_sha256": _sha256(partial),
            "resolved_model_revision": _resolved_revision(model),
            "versions": _versions(),
            "dtype": "float32",
            "torch_threads": threads,
        }
        # Metadata is written last.  Interrupted two-file commits are rejected
        # on the next read rather than being mistaken for a completed cache.
        os.replace(partial, path)
        _write_json(metadata_path, metadata)
        return _load_verified(path, expected)
    finally:
        if storage is not None:
            _close_memmap(storage)
        partial.unlink(missing_ok=True)
