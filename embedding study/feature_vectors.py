"""Normalize the feature blocks, apply weights and join them into A/B/C/D."""
import math
import json
import hashlib
from pathlib import Path
import numpy as np
from file_helpers import file_sha256, fingerprint, write_npy, write_json, package_versions
from file_helpers import file_sha256 as _sha256, read_json as _read_json

log = print
COMBINATIONS = {
    "A": {"categorical": 1.0, "description": 0.0, "popularity": 0.0},
    "B": {"categorical": 6 / 7, "description": 0.0, "popularity": 1 / 7},
    "C": {"categorical": 2 / 3, "description": 1 / 3, "popularity": 0.0},
    "D": {"categorical": 0.6, "description": 0.3, "popularity": 0.1},
}
POPULARITY_LABELS = (
    "Zero (0)", "Very low (1-10)", "Low (11-100)",
    "Moderate (101-1,000)", "High (1,001-10,000)",
    "Very high (10,001-100,000)", "Extreme (100,001+)",
)
REQUIRED_COLUMNS = ("AppID", "Name", "Description", "Peak CCU", "Popularity Bin")
SCHEMA_VERSION = 1


def normalize_rows(values):
    """Divide each game's vector by its length. A zero row stays zero."""
    numbers = np.asarray(values, dtype=np.float32)
    if numbers.ndim != 2 or not np.isfinite(numbers).all():
        raise ValueError("Normalization needs a two-dimensional table of finite numbers.")

    # float64 is used only for the length calculation, to avoid rounding errors.
    lengths = np.sqrt(np.sum(numbers.astype(np.float64) ** 2, axis=1))
    lengths[lengths == 0] = 1  # Dividing a zero row by 1 leaves it unchanged.
    normalized = numbers.copy()
    normalized /= lengths[:, None]  # [:, None] gives one divisor per game.
    return normalized


def encode_popularity(labels):
    """Use the seven existing bins; do not calculate new bins from Peak CCU."""
    bin_values = {
        "Zero (0)": 0 / 6,
        "Very low (1-10)": 1 / 6,
        "Low (11-100)": 2 / 6,
        "Moderate (101-1,000)": 3 / 6,
        "High (1,001-10,000)": 4 / 6,
        "Very high (10,001-100,000)": 5 / 6,
        "Extreme (100,001+)": 6 / 6,
    }
    popularity = []
    for label in labels:
        if not isinstance(label, str) or label not in bin_values:
            raise ValueError(f"Unknown or missing Popularity Bin: {label!r}.")
        popularity.append(bin_values[label])
    return np.array(popularity, dtype=np.float32).reshape(-1, 1)


def check_feature_block(values, name, number_of_games=None):
    """Check one input block before combining it with another block."""
    if values is None:
        raise ValueError(f"The {name} values are required for this combination.")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
        raise ValueError(f"The {name} block must be a finite two-dimensional table.")
    if number_of_games is not None and len(values) != number_of_games:
        raise ValueError("Feature blocks have different row counts.")
    if name == "popularity":
        if values.shape[1] != 1 or (values < 0).any() or (values > 1).any():
            raise ValueError("Popularity needs one column with values between 0 and 1.")
    return values


def combine_features(combination, categorical, descriptions=None, popularity=None):
    """Create A, B, C or D. Here categorical means tags + genres + categories."""
    if combination not in COMBINATIONS:
        raise ValueError("Choose combination A, B, C or D.")

    tags = check_feature_block(categorical, "tags")
    weights = COMBINATIONS[combination]
    tag_weight = np.float32(math.sqrt(weights["categorical"]))
    description_weight = np.float32(math.sqrt(weights["description"]))
    popularity_weight = np.float32(math.sqrt(weights["popularity"]))

    # Square roots make the weights apply to squared Euclidean distance.
    weighted_tags = tags * tag_weight
    if combination == "A":
        result = weighted_tags

    elif combination == "B":
        popularity = check_feature_block(popularity, "popularity", len(tags))
        result = np.concatenate([weighted_tags, popularity * popularity_weight], axis=1)

    elif combination == "C":
        descriptions = check_feature_block(descriptions, "description", len(tags))
        result = np.concatenate([weighted_tags, descriptions * description_weight], axis=1)

    else:  # D = tags + description + popularity, in this order.
        descriptions = check_feature_block(descriptions, "description", len(tags))
        popularity = check_feature_block(popularity, "popularity", len(tags))
        result = np.concatenate([
            weighted_tags,
            descriptions * description_weight,
            popularity * popularity_weight,
        ], axis=1)

    # Do not normalize again after joining the blocks.
    return np.ascontiguousarray(result, dtype=np.float32)


def feature_settings(combination, data, description_meta):
    weights = COMBINATIONS[combination]
    model_identity = None
    if weights["description"]:
        model_identity = {key: description_meta.get(key) for key in (
            "model_id", "model_revision", "resolved_model_revision", "max_seq_length", "pooling",
            "description_text_sha256", "app_ids_sha256", "dimension", "embedding_sha256", "cleaning_version")}
    return {
        "schema_version": SCHEMA_VERSION, "combination": combination,
        "source_signature": data["source_signature"], "weights": weights,
        "categorical_preprocessing": "existing weights; row L2 normalization; no second positional decay",
        "popularity_mapping": {label: i / 6 for i, label in enumerate(POPULARITY_LABELS)} if weights["popularity"] else None,
        "description": model_identity, "fusion": "sqrt block weights; no final normalization",
    }


def create_or_load_features(combination, data, categorical, descriptions, description_meta, output_dir, force):
    directory = output_dir / combination
    path, manifest_path = directory / "features.npy", directory / "features.json"
    identity = feature_settings(combination, data, description_meta)
    expected = fingerprint(identity)
    if path.exists() and manifest_path.exists() and not force:
        metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") != expected:
            raise ValueError(f"Existing {combination} features have different provenance. Use --force or a new output directory.")
        if metadata.get("feature_sha256") != file_sha256(path):
            raise ValueError(f"Existing {combination} feature file failed its checksum; rerun with --force.")
        features = np.load(path, mmap_mode="r", allow_pickle=False)
        if features.shape != (metadata["rows"], metadata["dimension"]):
            raise ValueError(f"Existing {combination} feature matrix shape does not match its manifest.")
        log(f"{combination}: reusing {features.shape} feature matrix")
        return features, metadata
    popularity = encode_popularity(data["frame"]["Popularity Bin"])
    features = combine_features(combination, categorical, descriptions, popularity)
    write_npy(path, features)
    metadata = {**identity, "fingerprint": expected, "rows": features.shape[0],
                "dimension": features.shape[1], "dtype": "float32", "feature_sha256": file_sha256(path),
                "versions": package_versions()}
    write_json(manifest_path, metadata)
    del features
    features = np.load(path, mmap_mode="r", allow_pickle=False)
    log(f"{combination}: saved {features.shape} feature matrix")
    return features, metadata


def read_features(root, combination, appids):
    """Load one saved matrix and check its game order and file contents."""
    feature_path = root / combination / "features.npy"
    metadata_path = root / combination / "features.json"
    metadata = _read_json(metadata_path)
    fingerprint = metadata.get("fingerprint")
    if not isinstance(fingerprint, str) or not fingerprint:
        raise ValueError(f"Missing feature fingerprint in {metadata_path}.")
    if not feature_path.is_file() or _sha256(feature_path) != metadata.get("feature_sha256"):
        raise ValueError(f"Missing or changed feature matrix {feature_path}; rebuild the combination before evaluation.")
    features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
    if features.ndim != 2 or features.shape[0] != len(appids):
        raise ValueError(f"Feature shape {features.shape} does not align with game_index.csv in {feature_path}.")
    if metadata.get("rows") != features.shape[0] or metadata.get("dimension") != features.shape[1]:
        raise ValueError(f"Feature dimensions disagree with metadata in {metadata_path}.")
    source_signature = metadata.get("source_signature", {})
    if isinstance(source_signature, dict) and source_signature.get("app_ids_sha256"):
        try:
            ordered_ids = np.asarray([int(appid) for appid in appids], dtype="<i8")
        except (ValueError, OverflowError) as exc:
            raise ValueError(f"game_index.csv must contain integer AppIDs for {metadata_path}.") from exc
        if hashlib.sha256(ordered_ids.tobytes()).hexdigest() != source_signature["app_ids_sha256"]:
            raise ValueError(f"Feature AppID provenance disagrees with game_index.csv in {metadata_path}.")
    if not np.issubdtype(features.dtype, np.floating):
        raise ValueError(f"Expected floating-point features in {feature_path}.")
    for start in range(0, len(appids), 8192):
        if not np.isfinite(features[start:start + 8192]).all():
            raise ValueError(f"Nonfinite values found in {feature_path}.")
    return features, metadata
