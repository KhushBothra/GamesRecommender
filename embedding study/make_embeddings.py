"""Make the four embeddings. Run: python "embedding study/make_embeddings.py" --offline"""

import argparse
import json
from pathlib import Path

from description_vectors import prepare_text, prepare_descriptions
from save_csv import export_one
from file_helpers import PROJECT_ROOT, STUDY_ROOT, file_sha256, write_json
from read_data import load_inputs, save_game_index
from feature_vectors import normalize_rows, create_or_load_features


def load_official_inputs(cleaned, descriptors):
    """Read Khush's two files. Their existing weights are used as supplied."""
    for path in (Path(cleaned), Path(descriptors)):
        with path.open("rb") as file:
            beginning = file.read(100).lstrip(b"\xef\xbb\xbf")
        if beginning.startswith(b"version https://git-lfs.github.com/spec/"):
            raise ValueError(f"{path} is an LFS pointer. Run git lfs pull first.")

    data = load_inputs(cleaned, descriptors)
    prepare_text(data["frame"])  # Check descriptions without loading MiniLM.
    return data


def export_or_reuse(root, combination, games, feature_info):
    """Keep an existing CSV only if its saved checks still match."""
    folder = root / combination
    csv_path = folder / "embeddings.csv"
    info_path = folder / "embeddings_csv_manifest.json"
    if not csv_path.exists() or not info_path.exists():
        return export_one(root, combination, games, overwrite=False)

    info = json.loads(info_path.read_text(encoding="utf-8"))
    correct_matrix = info.get("source_feature_sha256") == feature_info["feature_sha256"]
    correct_rows = info.get("rows") == len(games)
    correct_columns = info.get("feature_columns") == feature_info["dimension"]
    was_checked = info.get("every_coordinate_matches_float32_source") is True
    unchanged_csv = info.get("csv_sha256") == file_sha256(csv_path)
    if not all([correct_matrix, correct_rows, correct_columns, was_checked, unchanged_csv]):
        raise ValueError(
            f"{combination}: existing CSV failed validation. "
            "Use save_csv.py --overwrite to rebuild it."
        )
    print(f"{combination}: existing CSV is correct; keeping it.", flush=True)
    return info


def save_report(output_folder, data, sources, exports, plan):
    """Write the small summary files after all four combinations are ready."""
    plan["status"] = "complete"
    write_json(output_folder / "experiment_plan.json", plan)
    if not (output_folder / "run_summary.json").exists():
        write_json(output_folder / "run_summary.json", {
            "plan": plan, "completed_fits": 0, "runs": [],
        })
    write_json(output_folder / "embedding_csv_exports.json", {
        "sources": sources, "exports": exports,
    })

    matrices = []
    fields = ["combination", "rows", "feature_columns", "total_columns",
              "source_feature_sha256", "csv_sha256", "every_coordinate_matches_float32_source"]
    for export in exports:
        matrix_info = {}
        for field in fields:
            matrix_info[field] = export[field]
        matrices.append(matrix_info)
    write_json(output_folder.parent / "validation" / "embeddings_validation.json", {
        "status": "passed", "stage": "embeddings_only", "games": len(data["frame"]),
        "source_cleaned_sha256": sources["cleaned_csv_sha256"],
        "source_descriptors_sha256": sources["descriptors_csv_sha256"],
        "matrices": matrices, "models_trained": 0,
    })


def main(argv=None):
    # 1. Read settings. The usual command works with all these defaults.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned", type=Path, default=PROJECT_ROOT / "Data Prep/games_cleaned.csv")
    parser.add_argument("--descriptors", type=Path, default=PROJECT_ROOT / "Data Prep/games_descriptors.csv")
    parser.add_argument("--output-dir", type=Path, default=STUDY_ROOT / "outputs")
    parser.add_argument("--cache-dir", type=Path, default=STUDY_ROOT / "cache")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--prepare-only", action="store_true", help="Check inputs without creating outputs.")
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.threads < 1:
        parser.error("Batch size and threads must be positive.")

    # 2. Read the files and line up the games using AppID.
    data = load_official_inputs(args.cleaned, args.descriptors)
    games = data["frame"]
    print(f"Read {len(games):,} games and {len(data['descriptor_columns'])} weighted descriptors.")
    if args.prepare_only:
        print("Inputs valid. No source files or outputs changed.")
        return

    args.output_dir = args.output_dir.resolve()
    args.cache_dir = args.cache_dir.resolve()
    save_game_index(args.output_dir, data, sample_size=5000, sample_seed=42, force=False)
    sources = {
        "source": "official Data Prep CSVs; read-only; no descriptor recalculation",
        "cleaned_csv": str(args.cleaned.resolve()),
        "descriptors_csv": str(args.descriptors.resolve()),
        "cleaned_csv_sha256": data["source_signature"]["cleaned_sha256"],
        "descriptors_csv_sha256": data["source_signature"]["descriptors_sha256"],
    }
    write_json(args.output_dir / "descriptor_source.json", sources)
    plan = {
        "stage": "embeddings_only", "status": "running",
        "combinations": ["A", "B", "C", "D"], "rows": len(games),
        "source_rows": len(games), "subset_experiment": False,
        "cluster_counts": [], "seeds": [], "configurations": 4, "fits_including_seeds": 0,
        "source_signature": data["source_signature"],
    }
    write_json(args.output_dir / "experiment_plan.json", plan)

    # 3. Normalize the existing tag/genre/category values.
    tags = normalize_rows(data["categorical"])
    del data["categorical"]  # Release the unnormalized copy from memory.

    # 4. Make A, B, C and D. Only C and D need description vectors.
    descriptions = None
    description_info = {}
    exports = []
    for combination in ["A", "B", "C", "D"]:
        if combination in ["C", "D"] and descriptions is None:
            descriptions, description_info = prepare_descriptions(games, args)

        matrix, info = create_or_load_features(
            combination, data, tags, descriptions, description_info,
            args.output_dir, force=False,
        )
        del matrix

        # 5. Save a readable CSV, or keep the existing verified CSV.
        export = export_or_reuse(args.output_dir, combination, games, info)
        exports.append(export)

    # 6. Write a summary. This command creates embeddings, not KMeans models.
    save_report(args.output_dir, data, sources, exports, plan)
    print(f"Four verified embedding matrices and CSV exports ready: {args.output_dir}")


if __name__ == "__main__":
    main()
