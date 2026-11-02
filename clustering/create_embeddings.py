"""Create A/B/C/D embeddings from the two official Data Prep CSVs.

Run: python -m clustering.create_embeddings --offline
The source CSVs are read-only. Existing verified matrices and CSVs are reused.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .embeddings import _prepared_input, prepare_descriptions
from .export_embeddings import export_one
from .pipeline import (
    COMBINATIONS, PROJECT_ROOT, _prepare_features, _prepare_shared_outputs,
    build_parser, file_sha256, load_inputs, normalize_rows, write_json,
)


def load_official_inputs(cleaned, descriptors):
    """Read preweighted descriptors as supplied; never generate or edit them."""
    for path in (Path(cleaned), Path(descriptors)):
        with path.open("rb") as stream:
            if stream.read(100).lstrip(b"\xef\xbb\xbf").startswith(b"version https://git-lfs.github.com/spec/"):
                raise ValueError(f"{path} is an LFS pointer. Run git lfs pull first.")
    data = load_inputs(cleaned, descriptors)
    _prepared_input(data["frame"])  # Validate every cleaned description, without encoding.
    return data


def export_or_reuse(root, combination, frame, feature_meta):
    directory = root / combination
    path, manifest = directory / "embeddings.csv", directory / "embeddings_csv_manifest.json"
    if path.exists() and manifest.exists():
        record = json.loads(manifest.read_text(encoding="utf-8"))
        if (record.get("source_feature_sha256") != feature_meta["feature_sha256"]
                or record.get("rows") != len(frame)
                or record.get("feature_columns") != feature_meta["dimension"]
                or record.get("every_coordinate_matches_float32_source") is not True
                or record.get("csv_sha256") != file_sha256(path)):
            raise ValueError(f"{combination}: existing CSV failed validation; use clustering.export_embeddings --overwrite to rebuild it.")
        print(f"{combination}: reusing verified CSV without rewriting it", flush=True)
        return record
    return export_one(root, combination, frame, overwrite=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=PROJECT_ROOT / "Data Prep" / "games_cleaned.csv")
    parser.add_argument("--descriptors", type=Path, default=PROJECT_ROOT / "Data Prep" / "games_descriptors.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "clustering" / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "clustering" / "cache")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true", help="Validate both official inputs without writing files or encoding.")
    options = parser.parse_args(argv)
    if options.batch_size < 1 or options.threads < 1:
        parser.error("Batch size and threads must be positive.")
    data = load_official_inputs(options.cleaned, options.descriptors)
    print(f"Read {len(data['frame']):,} games and {len(data['descriptor_columns'])} preweighted descriptors from Data Prep.", flush=True)
    if options.prepare_only:
        print("Inputs valid. No source files or outputs changed.", flush=True)
        return

    args = build_parser().parse_args([])
    for name in ("cleaned", "descriptors", "output_dir", "cache_dir", "offline", "batch_size", "threads"):
        setattr(args, name, getattr(options, name))
    args.output_dir, args.cache_dir = args.output_dir.resolve(), args.cache_dir.resolve()
    _prepare_shared_outputs(args.output_dir, data, 5000, 42, False)
    sources = {
        "source": "official Data Prep CSVs; read-only; no descriptor recalculation",
        "cleaned_csv": str(options.cleaned.resolve()),
        "descriptors_csv": str(options.descriptors.resolve()),
        "cleaned_csv_sha256": data["source_signature"]["cleaned_sha256"],
        "descriptors_csv_sha256": data["source_signature"]["descriptors_sha256"],
    }
    write_json(args.output_dir / "descriptor_source.json", sources)
    plan = {
        "stage": "embeddings_only", "status": "running",
        "combinations": list(COMBINATIONS), "rows": len(data["frame"]),
        "source_rows": len(data["frame"]), "subset_experiment": False,
        "cluster_counts": [], "seeds": [], "configurations": 4, "fits_including_seeds": 0,
        "source_signature": data["source_signature"],
    }
    write_json(args.output_dir / "experiment_plan.json", plan)
    categorical = normalize_rows(data.pop("categorical"))
    descriptions, description_meta = None, {}
    exports = []
    for combination in COMBINATIONS:
        if COMBINATIONS[combination]["weights"]["description"] and descriptions is None:
            descriptions, description_meta = prepare_descriptions(data["frame"], args)
        features, metadata = _prepare_features(
            combination, data, categorical, descriptions, description_meta, args.output_dir, False)
        del features
        exports.append(export_or_reuse(args.output_dir, combination, data["frame"], metadata))
    plan["status"] = "complete"
    write_json(args.output_dir / "experiment_plan.json", plan)
    if not (args.output_dir / "run_summary.json").exists():
        write_json(args.output_dir / "run_summary.json", {"plan": plan, "completed_fits": 0, "runs": []})
    write_json(args.output_dir / "embedding_csv_exports.json", {"sources": sources, "exports": exports})
    write_json(PROJECT_ROOT / "clustering" / "validation" / "embeddings_validation.json", {
        "status": "passed", "stage": "embeddings_only", "games": len(data["frame"]),
        "source_cleaned_sha256": sources["cleaned_csv_sha256"],
        "source_descriptors_sha256": sources["descriptors_csv_sha256"],
        "matrices": [{key: row[key] for key in ("combination", "rows", "feature_columns", "total_columns", "source_feature_sha256", "csv_sha256", "every_coordinate_matches_float32_source")} for row in exports],
        "models_trained": 0,
    })
    print(f"Four verified embedding matrices and CSV exports ready: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
