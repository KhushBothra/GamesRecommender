"""Deferred, conventional-clustering metrics for saved experiment outputs.

This evaluates flat KMeans partitions, not overlapping Mapper clusters.  The
training entry points do not need to import this module or calculate metrics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from importlib import metadata as package_metadata
import json
import math
import platform
import re
from datetime import datetime, timezone
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Iterable


COMBINATIONS = ("A", "B", "C", "D")
SAMPLE_SEED = 42
METRIC_COLUMNS = (
    "combination", "n_clusters", "random_seed", "rows", "dimension",
    "actual_clusters", "sample_rows", "sample_clusters", "status", "reason",
    "davies_bouldin", "silhouette", "feature_fingerprint", "run_manifest",
)
SUMMARY_COLUMNS = (
    "combination", "n_clusters", "n_runs", "n_ok_runs", "n_invalid_runs",
    "davies_bouldin_mean", "davies_bouldin_std", "davies_bouldin_valid_runs",
    "silhouette_mean", "silhouette_std", "silhouette_valid_runs",
)


def _dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import numpy as np
        from sklearn import config_context
        from sklearn.metrics import davies_bouldin_score, silhouette_score
        from threadpoolctl import threadpool_limits
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation requires numpy, scikit-learn, and threadpoolctl. "
            "Install the clustering requirements with the same Python interpreter "
            "used to run this command, then rerun evaluation."
        ) from exc
    return np, config_context, davies_bouldin_score, silhouette_score, threadpool_limits


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read JSON metadata {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return data


def _read_csv(path: Path, required: Iterable[str]) -> list[dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            if len(reader.fieldnames or ()) != len(set(reader.fieldnames or ())):
                raise ValueError(f"Duplicate CSV column names in {path}.")
            missing = set(required) - set(reader.fieldnames or ())
            if missing:
                raise ValueError(f"Missing columns {sorted(missing)} in {path}.")
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Cannot read {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"No data rows in {path}.")
    if any(None in row or any(row.get(column) is None for column in required) for row in rows):
        raise ValueError(f"Malformed CSV rows in {path}.")
    return rows


def _validate_ids(rows: list[dict[str, str]], path: Path) -> list[str]:
    appids = [row.get("AppID", "") for row in rows]
    if any(not appid or appid != appid.strip() for appid in appids):
        raise ValueError(f"Empty or whitespace-padded AppID in {path}.")
    if len(set(appids)) != len(appids):
        raise ValueError(f"Duplicate AppIDs in {path}.")
    return appids


def _write_csv(path: Path, columns: Iterable[str], rows: Iterable[dict[str, Any]]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _integer_list(values: Iterable[int] | None, label: str, minimum: int) -> list[int] | None:
    if values is None:
        return None
    result = list(values)
    if not result or any(isinstance(value, bool) or not isinstance(value, int) or value < minimum for value in result):
        raise ValueError(f"{label} must contain integers >= {minimum}.")
    if len(set(result)) != len(result):
        raise ValueError(f"{label} must not contain duplicates.")
    return sorted(result)


def _selected_runs(
    root: Path,
    combinations: Iterable[str] | None,
    cluster_counts: Iterable[int] | None,
    seeds: Iterable[int] | None,
) -> list[tuple[str, int, int, Path]]:
    selected = list(combinations) if combinations is not None else [
        name for name in COMBINATIONS if (root / name / "features.json").is_file()
    ]
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(COMBINATIONS):
        raise ValueError("Select existing feature combinations from A, B, C, D without duplicates.")
    counts = _integer_list(cluster_counts, "Cluster counts", 2)
    selected_seeds = _integer_list(seeds, "Seeds", 0)
    runs = []
    for combination in selected:
        directory = root / combination
        available_counts = sorted(
            int(match.group(1))
            for path in directory.glob("k_*")
            if path.is_dir() and (match := re.fullmatch(r"k_(\d+)", path.name))
        )
        chosen_counts = counts if counts is not None else available_counts
        if not chosen_counts:
            raise ValueError(f"No clustering runs found for combination {combination} in {directory}.")
        for count in chosen_counts:
            count_dir = directory / f"k_{count:02d}"
            available_seeds = sorted(
                int(match.group(1))
                for path in count_dir.glob("seed_*")
                if path.is_dir() and (match := re.fullmatch(r"seed_(\d+)", path.name))
            )
            chosen_seeds = selected_seeds if selected_seeds is not None else available_seeds
            if not chosen_seeds:
                raise ValueError(f"No seed runs found in {count_dir}; train this configuration first.")
            for seed in chosen_seeds:
                manifest_path = count_dir / f"seed_{seed}" / "manifest.json"
                if not manifest_path.is_file():
                    raise ValueError(f"Missing run manifest {manifest_path}; finish training this run first.")
                runs.append((combination, count, seed, manifest_path))
    return runs


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    for combination, count in sorted({(row["combination"], row["n_clusters"]) for row in rows}):
        group = [row for row in rows if row["combination"] == combination and row["n_clusters"] == count]
        record = {
            "combination": combination,
            "n_clusters": count,
            "n_runs": len(group),
            "n_ok_runs": sum(row["status"] == "ok" for row in group),
            "n_invalid_runs": sum(row["status"] != "ok" for row in group),
        }
        for metric in ("davies_bouldin", "silhouette"):
            values = [row[metric] for row in group if isinstance(row[metric], (int, float)) and math.isfinite(row[metric])]
            record[metric + "_mean"] = fmean(values) if values else ""
            record[metric + "_std"] = pstdev(values) if values else ""
            record[metric + "_valid_runs"] = len(values)
        summary.append(record)
    return summary


def evaluate_outputs(
    output_dir: str | Path,
    combinations: Iterable[str] | None = None,
    cluster_counts: Iterable[int] | None = None,
    seeds: Iterable[int] | None = None,
    sample_size: int | None = None,
) -> list[dict[str, Any]]:
    """Verify saved artifacts and write DBI/silhouette CSVs and provenance.

    Without ``sample_size`` the training-time evaluation_sample_appids.csv is
    reused.  An override selects min(sample_size, catalogue size) IDs with seed
    42, once for every selected fit.  Results use Euclidean geometry directly
    from the saved, weighted features; no additional normalization is applied.
    Invalid partitions are explicitly marked, never assigned a healthy score.
    """
    np, config_context, db_score, silhouette_score, threadpool_limits = _dependencies()
    root = Path(output_dir).expanduser().resolve()
    index_path = root / "game_index.csv"
    game_index = _read_csv(index_path, ("AppID", "Name"))
    appids = _validate_ids(game_index, index_path)
    if len(appids) < 3:
        raise ValueError("Evaluation needs at least three games and a nontrivial partition.")
    id_to_position = {appid: index for index, appid in enumerate(appids)}
    if sample_size is None:
        sample_source = root / "evaluation_sample_appids.csv"
        sample_rows = _read_csv(sample_source, ("AppID",))
        sample_ids = _validate_ids(sample_rows, sample_source)
        sample_method = "saved training-time sample"
    else:
        if isinstance(sample_size, bool) or not isinstance(sample_size, int) or sample_size < 2:
            raise ValueError("sample_size must be an integer >= 2.")
        sample_positions = np.sort(np.random.default_rng(SAMPLE_SEED).choice(
            len(appids), size=min(sample_size, len(appids)), replace=False,
        ))
        sample_ids = [appids[int(position)] for position in sample_positions]
        sample_source = None
        sample_method = "numpy.default_rng(42), without replacement, catalogue row order"
    unknown = set(sample_ids) - set(appids)
    if unknown:
        raise ValueError(f"Evaluation sample contains AppIDs absent from game_index.csv: {sorted(unknown)[:5]}.")
    if len(sample_ids) < 2:
        raise ValueError("The evaluation sample must contain at least two games.")
    sample_indices = np.asarray([id_to_position[appid] for appid in sample_ids], dtype=np.int64)
    selected_runs = _selected_runs(root, combinations, cluster_counts, seeds)
    rows: list[dict[str, Any]] = []
    feature_provenance: dict[str, dict[str, Any]] = {}
    source_manifests: dict[str, str] = {}

    # Process one combination at a time, holding only one catalogue matrix and
    # its sampled rows.  sklearn silhouette uses bounded pairwise chunks.
    for combination in dict.fromkeys(run[0] for run in selected_runs):
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
        sample_features = features[sample_indices]
        feature_provenance[combination] = {
            "fingerprint": fingerprint,
            "feature_sha256": metadata["feature_sha256"],
            "metadata_sha256": _sha256(metadata_path),
            "rows": int(features.shape[0]),
            "dimension": int(features.shape[1]),
        }
        for _, count, seed, manifest_path in (run for run in selected_runs if run[0] == combination):
            manifest = _read_json(manifest_path)
            if manifest.get("status") != "complete":
                raise ValueError(f"Run is not complete: {manifest_path}. Finish or rerun training first.")
            if (manifest.get("combination"), manifest.get("n_clusters"), manifest.get("random_seed")) != (combination, count, seed):
                raise ValueError(f"Run identity disagrees with its directory: {manifest_path}.")
            if manifest.get("feature_fingerprint") != fingerprint:
                raise ValueError(f"Run uses a different feature fingerprint: {manifest_path}; retrain this run.")
            if "feature_sha256" in manifest and manifest["feature_sha256"] != metadata["feature_sha256"]:
                raise ValueError(f"Run feature checksum disagrees with the feature matrix: {manifest_path}.")
            for key, expected in (("rows", len(appids)), ("dimension", int(features.shape[1]))):
                if key in manifest and manifest[key] != expected:
                    raise ValueError(f"Run {key} disagrees with the feature matrix: {manifest_path}.")
            files = manifest.get("files")
            if not isinstance(files, dict):
                raise ValueError(f"Missing artifact hashes in {manifest_path}.")
            for filename in ("centers.npy", "labels.csv"):
                artifact = manifest_path.parent / filename
                if not artifact.is_file() or _sha256(artifact) != files.get(filename):
                    raise ValueError(f"Missing or changed artifact {artifact}; retrain this run.")
            labels_path = manifest_path.parent / "labels.csv"
            label_rows = _read_csv(labels_path, ("AppID", "Name", "cluster"))
            if _validate_ids(label_rows, labels_path) != appids:
                raise ValueError(f"AppID row order differs from game_index.csv in {labels_path}.")
            if any(label_row["Name"] != index_row["Name"] for label_row, index_row in zip(label_rows, game_index)):
                raise ValueError(f"Game names differ from game_index.csv in {labels_path}.")
            try:
                labels = np.asarray([int(row["cluster"]) for row in label_rows], dtype=np.int64)
            except (TypeError, ValueError, OverflowError) as exc:
                raise ValueError(f"Noninteger cluster label in {labels_path}.") from exc
            if (labels < 0).any() or (labels >= count).any():
                raise ValueError(f"Cluster labels outside [0, {count - 1}] in {labels_path}.")
            centers = np.load(manifest_path.parent / "centers.npy", allow_pickle=False)
            if centers.shape != (count, features.shape[1]) or not np.isfinite(centers).all():
                raise ValueError(f"Invalid centre shape or values in {manifest_path.parent / 'centers.npy'}.")
            actual_clusters = int(np.unique(labels).size)
            if "actual_clusters" in manifest and manifest["actual_clusters"] != actual_clusters:
                raise ValueError(f"Run actual_clusters disagrees with saved labels: {manifest_path}.")
            sample_labels = labels[sample_indices]
            sample_clusters = int(np.unique(sample_labels).size)
            relative_manifest = manifest_path.relative_to(root).as_posix()
            source_manifests[relative_manifest] = _sha256(manifest_path)
            record = {
                "combination": combination,
                "n_clusters": count,
                "random_seed": seed,
                "rows": len(appids),
                "dimension": int(features.shape[1]),
                "actual_clusters": actual_clusters,
                "sample_rows": len(sample_ids),
                "sample_clusters": sample_clusters,
                "status": "ok",
                "reason": "",
                "davies_bouldin": "",
                "silhouette": "",
                "feature_fingerprint": fingerprint,
                "run_manifest": relative_manifest,
            }
            reasons = []
            if actual_clusters != count or not 2 <= actual_clusters < len(appids):
                record["status"] = "degenerate_partition"
                reasons.append(f"Requested {count} clusters, observed {actual_clusters} across {len(appids)} games")
            elif np.unique(centers, axis=0).shape[0] != count:
                record["status"] = "degenerate_partition"
                reasons.append("Distinct cluster labels have duplicate saved centres")
            else:
                with threadpool_limits(limits=1), config_context(working_memory=64):
                    dbi = float(db_score(features, labels))
                    if math.isfinite(dbi):
                        record["davies_bouldin"] = dbi
                    else:
                        record["status"] = "invalid_metric"
                        reasons.append("Davies-Bouldin score is nonfinite")
                    if 2 <= sample_clusters < len(sample_ids):
                        silhouette = float(silhouette_score(sample_features, sample_labels, metric="euclidean"))
                        if math.isfinite(silhouette):
                            record["silhouette"] = silhouette
                        else:
                            record["status"] = "invalid_metric"
                            reasons.append("Silhouette score is nonfinite")
                    else:
                        record["status"] = "degenerate_sample"
                        reasons.append(f"Sample has {sample_clusters} clusters across {len(sample_ids)} games; use a larger fixed sample")
            record["reason"] = "; ".join(reasons)
            rows.append(record)
            print(
                f"Evaluated {combination}, k={count}, seed={seed}: {record['status']}; "
                f"DBI={record['davies_bouldin']}, silhouette={record['silhouette']}",
                flush=True,
            )
        del features, sample_features

    sample_output = root / "metrics_sample_appids.csv"
    _write_csv(sample_output, ("AppID",), ({"AppID": appid} for appid in sample_ids))
    _write_csv(root / "metrics.csv", METRIC_COLUMNS, rows)
    _write_csv(root / "metrics_summary.csv", SUMMARY_COLUMNS, _summary(rows))
    _write_json(root / "metrics_manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            **{name: package_metadata.version(name) for name in ("numpy", "scikit-learn", "threadpoolctl")},
        },
        "evaluation_type": "flat clustering; not Mapper graph evaluation",
        "metrics": {
            "davies_bouldin": "full catalogue, Euclidean geometry; lower is better",
            "silhouette": "fixed sampled AppIDs, Euclidean geometry; higher is better",
            "pairwise_working_memory_mib": 64,
            "threadpool_limit": 1,
        },
        "aggregation": "per combination and k, arithmetic mean and population standard deviation across valid seed scores; single-score std is 0",
        "sample": {
            "method": sample_method,
            "seed_for_override": SAMPLE_SEED,
            "requested_size_override": sample_size,
            "actual_size": len(sample_ids),
            "source_file": sample_source.name if sample_source is not None else None,
            "source_sha256": _sha256(sample_source) if sample_source is not None else None,
            "evaluated_ids_file": sample_output.name,
            "evaluated_ids_sha256": _sha256(sample_output),
            "appids": sample_ids,
        },
        "game_index_sha256": _sha256(index_path),
        "features": feature_provenance,
        "source_manifest_sha256": source_manifests,
        "runs_evaluated": len(rows),
        "runs_ok": sum(row["status"] == "ok" for row in rows),
        "outputs": {
            name: _sha256(root / name)
            for name in ("metrics.csv", "metrics_summary.csv")
        },
    })
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--combinations", nargs="+", choices=COMBINATIONS)
    parser.add_argument("--clusters", nargs="+", type=int, dest="cluster_counts")
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--sample-size", type=int, help="Override saved evaluation sample using a single deterministic seed-42 sample")
    arguments = parser.parse_args(argv)
    try:
        rows = evaluate_outputs(**vars(arguments))
    except (OSError, ValueError, RuntimeError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    invalid = sum(row["status"] != "ok" for row in rows)
    print(f"Saved metrics for {len(rows)} runs to {arguments.output_dir.resolve()} ({invalid} runs flagged).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
