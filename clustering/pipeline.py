"""Shared implementation for the four game clustering combinations.

Source CSVs are read only. Feature matrices, fitted estimators, AppID assignments,
and provenance are written below the selected output directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTER_COUNTS = (2, 3, 5, 8, 13, 21, 34)
COMBINATIONS = {
    "A": {"weights": {"categorical": 1.0, "description": 0.0, "popularity": 0.0}},
    "B": {"weights": {"categorical": 6 / 7, "description": 0.0, "popularity": 1 / 7}},
    "C": {"weights": {"categorical": 2 / 3, "description": 1 / 3, "popularity": 0.0}},
    "D": {"weights": {"categorical": 0.6, "description": 0.3, "popularity": 0.1}},
}
POPULARITY_LABELS = (
    "Zero (0)", "Very low (1-10)", "Low (11-100)",
    "Moderate (101-1,000)", "High (1,001-10,000)",
    "Very high (10,001-100,000)", "Extreme (100,001+)",
)
REQUIRED_COLUMNS = ("AppID", "Name", "Description", "Peak CCU", "Popularity Bin")
SCHEMA_VERSION = 1


def log(message):
    print(message, flush=True)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def app_ids_sha256(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def write_npy(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        np.save(stream, values, allow_pickle=False)
    temporary.replace(path)


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    frame.to_csv(temporary, index=False, encoding="utf-8")
    temporary.replace(path)


def package_versions():
    result = {"python": platform.python_version()}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "joblib", "sentence-transformers", "torch"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return result


def normalize_rows(values):
    """Return a float32 copy; do not mutate or reweight the supplied descriptors."""
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Normalization requires a finite two-dimensional numeric matrix.")
    result = array.copy()
    # Accumulate in float64 for stable lengths while storing features as float32.
    lengths = np.sqrt(np.sum(result.astype(np.float64) ** 2, axis=1))
    np.divide(result, lengths[:, None], out=result, where=lengths[:, None] > 0)
    return result


def encode_popularity(labels):
    mapping = {label: index / 6 for index, label in enumerate(POPULARITY_LABELS)}
    result = []
    for label in labels:
        if not isinstance(label, str) or label not in mapping:
            raise ValueError(f"Unknown or missing Popularity Bin: {label!r}. Expected the existing seven labels.")
        result.append(mapping[label])
    return np.asarray(result, dtype=np.float32).reshape(-1, 1)


def combine_features(combination, categorical, descriptions=None, popularity=None):
    """Concatenate normalized blocks using coefficients in squared distance.

    In particular, no global row normalization is applied after concatenation.
    """
    if combination not in COMBINATIONS:
        raise ValueError(f"Unknown combination {combination!r}; expected A, B, C, or D.")
    weights = COMBINATIONS[combination]["weights"]
    blocks = {"categorical": categorical, "description": descriptions, "popularity": popularity}
    prepared = []
    rows = None
    for name, weight in weights.items():
        if not weight:
            continue
        if blocks[name] is None:
            raise ValueError(f"Combination {combination} requires the {name} block.")
        values = np.asarray(blocks[name], dtype=np.float32)
        if values.ndim != 2 or values.shape[1] == 0 or not np.isfinite(values).all():
            raise ValueError(f"The {name} block must be a finite two-dimensional matrix.")
        if rows is None:
            rows = len(values)
        elif rows != len(values):
            raise ValueError("Feature blocks have different row counts.")
        if name == "popularity" and (values.shape[1] != 1 or (values < 0).any() or (values > 1).any()):
            raise ValueError("Popularity must have one column with values between zero and one.")
        prepared.append(values * np.float32(math.sqrt(weight)))
    return np.ascontiguousarray(np.concatenate(prepared, axis=1), dtype=np.float32)


def _csv_header(path):
    try:
        with Path(path).open(encoding="utf-8-sig", newline="") as stream:
            header = next(csv.reader(stream))
    except (OSError, StopIteration, csv.Error) as exc:
        raise ValueError(f"Cannot read CSV header from {path}: {exc}") from exc
    if len(header) != len(set(header)) or any(not label for label in header):
        raise ValueError(f"Duplicate or empty CSV column labels in {path}.")
    return header


def _validated_ids(series, path):
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
    clean_header, descriptor_header = _csv_header(cleaned_path), _csv_header(descriptor_path)
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
    frame["AppID"] = _validated_ids(frame["AppID"], cleaned_path)
    descriptors["AppID"] = _validated_ids(descriptors["AppID"], descriptor_path)
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


def _prepare_shared_outputs(output_dir, data, sample_size, sample_seed, force):
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


def _feature_identity(combination, data, description_meta):
    weights = COMBINATIONS[combination]["weights"]
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


def _prepare_features(combination, data, categorical, descriptions, description_meta, output_dir, force):
    directory = output_dir / combination
    path, manifest_path = directory / "features.npy", directory / "features.json"
    identity = _feature_identity(combination, data, description_meta)
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


def _fit_one(combination, features, feature_meta, frame, k, seed, args):
    try:
        import joblib
        from sklearn.cluster import KMeans
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise ValueError("Clustering dependencies are missing. Install clustering/requirements-core.txt in your Python environment.") from exc
    directory = args.output_dir / combination / f"k_{k:02d}" / f"seed_{seed}"
    manifest_path = directory / "manifest.json"
    params = {"n_clusters": k, "init": "k-means++", "n_init": args.n_init,
              "max_iter": args.max_iter, "tol": args.tol, "random_state": seed, "algorithm": "lloyd"}
    fit_versions = {key: value for key, value in package_versions().items()
                    if key in {"python", "numpy", "scikit-learn", "scipy", "joblib"}}
    identity = {"feature_fingerprint": feature_meta["fingerprint"], "feature_sha256": feature_meta["feature_sha256"],
                "parameters": params, "versions": fit_versions}
    expected = fingerprint(identity)
    if manifest_path.exists() and not args.force:
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("fingerprint") != expected:
            raise ValueError(f"Existing run {combination}/k={k}/seed={seed} has different parameters or features. Use --force or a new output directory.")
        if existing.get("status") == "complete":
            for filename, digest in existing["files"].items():
                if not (directory / filename).is_file() or file_sha256(directory / filename) != digest:
                    raise ValueError(f"Saved run {directory} failed artifact validation. Use --force to rebuild it.")
            log(f"{combination} k={k:2d} seed={seed}: complete, skipping")
            return existing
    directory.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    log(f"{combination} k={k:2d} seed={seed}: fitting KMeans, n_init={args.n_init}")
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=args.threads):
        warnings.simplefilter("always")
        model = KMeans(**params).fit(features)
    labels = np.asarray(model.labels_, dtype=np.int32)
    actual = len(np.unique(labels))
    if actual != k:
        raise ValueError(f"{combination} requested {k} clusters but produced {actual}; inspect duplicate vectors or clustering warnings.")
    if not np.isfinite(model.cluster_centers_).all():
        raise ValueError("Clustering produced nonfinite cluster centres.")
    model_path = directory / "model.joblib"
    temporary_model = model_path.with_name(model_path.name + ".partial")
    joblib.dump(model, temporary_model, compress=3)
    temporary_model.replace(model_path)
    write_npy(directory / "centers.npy", model.cluster_centers_)
    assignments = frame[["AppID", "Name"]].copy()
    assignments["cluster"] = labels
    write_csv(directory / "labels.csv", assignments)
    manifest = {
        **identity, "schema_version": SCHEMA_VERSION, "status": "complete", "fingerprint": expected,
        "combination": combination, "n_clusters": k, "actual_clusters": actual, "random_seed": seed,
        "rows": len(features), "dimension": features.shape[1], "inertia": float(model.inertia_),
        "iterations": int(model.n_iter_), "max_iter_reached": bool(model.n_iter_ >= args.max_iter),
        "cluster_sizes": np.bincount(labels, minlength=k).tolist(), "fit_seconds": round(time.perf_counter() - start, 3),
        "warnings": [str(item.message) for item in caught],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {name: file_sha256(directory / name) for name in ("model.joblib", "centers.npy", "labels.csv")},
    }
    write_json(manifest_path, manifest)  # Written last: a completed run is resumable.
    log(f"{combination} k={k:2d} seed={seed}: saved, iterations={model.n_iter_}, {manifest['fit_seconds']:.1f}s")
    return manifest


def build_parser(default_combinations=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--combinations", nargs="+", choices=tuple(COMBINATIONS), default=default_combinations or list(COMBINATIONS))
    parser.add_argument("--clusters", nargs="+", type=int, default=list(CLUSTER_COUNTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--cleaned", type=Path, default=PROJECT_ROOT / "Data Prep" / "games_cleaned.csv")
    parser.add_argument("--descriptors", type=Path, default=PROJECT_ROOT / "Data Prep" / "games_descriptors.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "clustering" / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "clustering" / "cache")
    parser.add_argument("--limit", type=int, help="Deterministic subset for smoke tests; omit for the full experiment.")
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--silhouette-sample-size", type=int, default=5000)
    parser.add_argument("--n-init", type=int, default=10)
    parser.add_argument("--max-iter", type=int, default=300)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--description-model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--model-revision", default=None)
    parser.add_argument("--max-seq-length", type=int, default=256)
    parser.add_argument("--chunk-descriptions", action="store_true")
    parser.add_argument("--description-embeddings", type=Path)
    parser.add_argument("--offline", action="store_true", help="Load the text model only from its local cache/path.")
    parser.add_argument("--force", action="store_true", help="Rebuild selected outputs instead of reusing them.")
    parser.add_argument("--dry-run", action="store_true", help="Validate CSVs and show the plan without creating outputs or downloading models.")
    parser.add_argument("--metrics", action="store_true", help="Optionally evaluate completed models; omitted by default.")
    return parser


def run(args):
    args.combinations = list(dict.fromkeys(args.combinations))
    args.clusters = list(dict.fromkeys(args.clusters))
    args.seeds = list(dict.fromkeys(args.seeds))
    for name in ("n_init", "max_iter", "threads", "batch_size", "max_seq_length", "silhouette_sample_size"):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if args.max_seq_length < 8 or args.tol <= 0 or not math.isfinite(args.tol):
        raise ValueError("Use max-seq-length >= 8 and a positive finite tolerance.")
    if any(k < 2 for k in args.clusters) or any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        raise ValueError("Cluster counts must be >=2 and random seeds must be between 0 and 2**32-1.")
    args.output_dir, args.cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    log("Reading and validating the two source CSVs...")
    data = load_inputs(args.cleaned, args.descriptors, args.limit, args.sample_seed)
    frame = data["frame"]
    if max(args.clusters) >= len(frame):
        raise ValueError("Each requested cluster count must be smaller than the number of selected games.")
    log(f"Validated {len(frame):,} games and {len(data['descriptor_columns'])} existing weighted descriptors.")
    plan = {
        "combinations": args.combinations, "cluster_counts": args.clusters, "seeds": args.seeds,
        "configurations": len(args.combinations) * len(args.clusters),
        "fits_including_seeds": len(args.combinations) * len(args.clusters) * len(args.seeds),
        "rows": len(frame), "source_rows": data["source_signature"]["source_rows"],
        "subset_experiment": args.limit is not None, "weights": {c: COMBINATIONS[c]["weights"] for c in args.combinations},
    }
    if args.dry_run:
        log(json.dumps(plan, indent=2))
        return plan
    _prepare_shared_outputs(args.output_dir, data, args.silhouette_sample_size, args.sample_seed, args.force)
    categorical = normalize_rows(data.pop("categorical"))
    descriptions, description_meta = None, {}
    manifests = []
    run_plan_path = args.output_dir / "experiment_plan.json"
    write_json(run_plan_path, {**plan, "status": "running", "source_signature": data["source_signature"]})
    for combination in args.combinations:
        if COMBINATIONS[combination]["weights"]["description"] and descriptions is None:
            from .embeddings import prepare_descriptions
            descriptions, description_meta = prepare_descriptions(frame, args)
        features, metadata = _prepare_features(combination, data, categorical, descriptions,
                                               description_meta, args.output_dir, args.force)
        for k in args.clusters:
            for seed in args.seeds:
                manifest = _fit_one(combination, features, metadata, frame, k, seed, args)
                manifests.append(manifest)
                write_json(args.output_dir / "run_summary.json", {
                    "plan": plan, "completed_fits": len(manifests),
                    "runs": [{key: m[key] for key in ("combination", "n_clusters", "random_seed", "inertia", "iterations", "max_iter_reached", "fit_seconds")} for m in manifests],
                })
        del features
    write_json(run_plan_path, {**plan, "status": "complete", "source_signature": data["source_signature"]})
    log(f"Completed {len(manifests)} fits across {plan['configurations']} configurations. Outputs: {args.output_dir}")
    if args.metrics:
        from .evaluate import evaluate_outputs
        evaluate_outputs(args.output_dir, args.combinations, args.clusters, args.seeds)
    return plan


def main(argv=None, default_combinations=None):
    parser = build_parser(default_combinations)
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, ImportError, RuntimeError, json.JSONDecodeError) as exc:
        parser.exit(2, f"Error: {exc}\n")
    except KeyboardInterrupt:
        parser.exit(130, "Interrupted. Completed model artifacts can be resumed with the same command.\n")


if __name__ == "__main__":
    main()
