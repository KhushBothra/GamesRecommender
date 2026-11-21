"""Read Khush's two CSV files and match rows using AppID."""
import csv
import json
from pathlib import Path
import numpy as np
import pandas as pd
from feature_vectors import REQUIRED_COLUMNS, encode_popularity
from file_helpers import file_sha256, app_ids_sha256, fingerprint, write_csv, write_npy, write_json
SCHEMA_VERSION = 1

def read_csv_columns(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
    except (OSError, StopIteration, csv.Error) as exc:
        raise ValueError(f"Cannot read CSV header from {path}: {exc}") from exc
    if len(header) != len(set(header)) or any(not label for label in header):
        raise ValueError(f"Duplicate or empty CSV column labels in {path}.")
    return header


def check_app_ids(series, path):
    try:
        numeric = pd.to_numeric(series, errors="raise")
        array = numeric.to_numpy(dtype=np.float64)
        if not np.isfinite(array).all() or (array <= 0).any() or (array != np.floor(array)).any():
            raise ValueError("AppIDs must be positive integers.")
        integers = numeric.astype(np.int64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"Invalid AppID in {path}: {exc}") from exc
    if integers.duplicated().any():
        raise ValueError(f"Duplicate AppIDs in {path}.")
    return integers


def load_inputs(cleaned_path, descriptor_path, limit=None, sample_seed=42):
    """Validate full populations first, align by AppID, then optionally sample."""
    cleaned_path, descriptor_path = Path(cleaned_path), Path(descriptor_path)
    clean_header, descriptor_header = read_csv_columns(cleaned_path), read_csv_columns(descriptor_path)
    missing = set(REQUIRED_COLUMNS) - set(clean_header)
    if missing or "AppID" not in descriptor_header:
        raise ValueError(f"Missing required columns: cleaned={sorted(missing)}, descriptors require AppID.")
    columns = [column for column in descriptor_header if column != "AppID"]
    if not columns:
        raise ValueError("The descriptor file contains no numerical features.")
    try:
        frame = pd.read_csv(cleaned_path, usecols=list(REQUIRED_COLUMNS), encoding="utf-8-sig")
        descriptors = pd.read_csv(descriptor_path, dtype={column: np.float32 for column in columns},
                                  encoding="utf-8-sig")
    except (ValueError, TypeError, OSError, pd.errors.ParserError) as exc:
        raise ValueError(f"Cannot parse the input CSVs: {exc}") from exc
    frame["AppID"] = check_app_ids(frame["AppID"], cleaned_path)
    descriptors["AppID"] = check_app_ids(descriptors["AppID"], descriptor_path)
    clean_ids, descriptor_ids = set(frame["AppID"]), set(descriptors["AppID"])
    if clean_ids != descriptor_ids:
        raise ValueError("The two CSVs contain different AppID populations; align the source files before training.")
    if not len(frame):
        raise ValueError("The input CSVs contain no games.")
    # Reject invalid source values even when a smoke-test subset is requested.
    source_values = descriptors[columns].to_numpy(dtype=np.float32)
    if not np.isfinite(source_values).all() or (source_values < 0).any() or (source_values > 1).any():
        raise ValueError("Existing descriptor weights must be finite numbers between zero and one.")
    encode_popularity(frame["Popularity Bin"].tolist())
    descriptors = descriptors.set_index("AppID").loc[frame["AppID"], columns]
    full_count = len(frame)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be a positive number of games.")
        if limit > len(frame):
            raise ValueError(f"--limit {limit} exceeds the available {len(frame)} games.")
        selected = np.sort(np.random.default_rng(sample_seed).choice(len(frame), size=limit, replace=False))
        frame = frame.iloc[selected]
        descriptors = descriptors.iloc[selected]
    frame = frame.reset_index(drop=True)
    categorical = np.ascontiguousarray(descriptors.to_numpy(dtype=np.float32))
    signature = {
        "cleaned_sha256": file_sha256(cleaned_path),
        "descriptors_sha256": file_sha256(descriptor_path),
        "descriptor_columns": columns,
        "source_rows": full_count, "selected_rows": len(frame),
        "app_ids_sha256": app_ids_sha256(frame["AppID"]),
        "limit": limit, "sample_seed": int(sample_seed),
    }
    return {"frame": frame, "categorical": categorical, "descriptor_columns": columns,
            "source_signature": signature}


def save_game_index(output_dir, data, sample_size, sample_seed, force):
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "data_manifest.json"
    identity = {"schema_version": SCHEMA_VERSION, "source_signature": data["source_signature"],
                "sample_size": min(sample_size, len(data["frame"])), "sample_seed": sample_seed}
    expected = fingerprint(identity)
    if identity_path.exists():
        old = json.loads(identity_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") != expected and not force:
            raise ValueError("Output directory belongs to different input data/sample settings. Use a new --output-dir or --force.")
    frame = data["frame"]
    write_csv(output_dir / "game_index.csv", frame[["AppID", "Name"]])
    write_npy(output_dir / "app_ids.npy", frame["AppID"].to_numpy(dtype=np.int64))
    selection = np.sort(np.random.default_rng(sample_seed).choice(len(frame), size=identity["sample_size"], replace=False))
    write_csv(output_dir / "evaluation_sample_appids.csv", frame.iloc[selection][["AppID"]])
    identity["fingerprint"] = expected
    write_json(identity_path, identity)
