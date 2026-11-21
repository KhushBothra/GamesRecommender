"""Export the saved embedding matrices to CSV without encoding games again."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from file_helpers import PROJECT_ROOT, app_ids_sha256, file_sha256, write_json


def column_names(info):
    """The column order must match combine_features(): tags, description, popularity."""
    names = []
    for tag in info["source_signature"]["descriptor_columns"]:
        names.append("categorical__" + tag)
    if info["weights"]["description"]:
        for number in range(info["description"]["dimension"]):
            names.append(f"description__{number:03d}")
    if info["weights"]["popularity"]:
        names.append("popularity__weighted")
    return names


def check_csv(csv_path, games, matrix, names):
    """Read the CSV back in small parts and compare every game and number."""
    types = {"AppID": np.int64, "Name": str}
    for name in names:
        types[name] = np.float32
    chunks = pd.read_csv(
        csv_path, chunksize=5000, dtype=types, keep_default_na=False,
        encoding="utf-8-sig", float_precision="round_trip",
    )
    checked = 0
    for chunk in chunks:
        stop = checked + len(chunk)
        expected_games = games.iloc[checked:stop]
        if chunk.columns.tolist() != ["AppID", "Name"] + names:
            raise ValueError("CSV column order does not match the matrix.")
        if not np.array_equal(chunk["AppID"].to_numpy(), expected_games["AppID"].to_numpy()):
            raise ValueError("CSV AppID order does not match the game index.")
        if chunk["Name"].tolist() != expected_games["Name"].tolist():
            raise ValueError("CSV game names do not match the game index.")
        numbers = chunk[names].to_numpy(dtype=np.float32)
        if not np.array_equal(numbers, matrix[checked:stop]):
            raise ValueError("CSV numbers do not match features.npy.")
        checked = stop
    if checked != len(games):
        raise ValueError("CSV row count does not match the game index.")
    return checked


def export_one(root, combination, index, overwrite=False):
    # 1. Load the matrix and its information file.
    games = index
    folder = root / combination
    info = json.loads((folder / "features.json").read_text(encoding="utf-8"))
    matrix_path = folder / "features.npy"
    if file_sha256(matrix_path) != info["feature_sha256"]:
        raise ValueError(f"{combination}: feature checksum mismatch.")
    matrix = np.load(matrix_path, mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(games), info["dimension"]):
        raise ValueError(f"{combination}: feature shape does not match the game index.")
    if app_ids_sha256(games["AppID"]) != info["source_signature"]["app_ids_sha256"]:
        raise ValueError(f"{combination}: AppID order does not match the saved matrix.")
    names = column_names(info)
    if len(names) != matrix.shape[1] or len(set(names)) != len(names):
        raise ValueError(f"{combination}: feature header mismatch.")

    csv_path = folder / "embeddings.csv"
    preview_path = folder / "embeddings_preview.csv"
    if csv_path.exists() and not overwrite:
        raise ValueError(f"{csv_path} already exists; use --overwrite to replace it.")
    temporary = csv_path.with_name(csv_path.name + ".partial")

    # 2. Write 5,000 games at a time, so the whole CSV need not fit in memory.
    print(f"{combination}: exporting {len(games):,} games", flush=True)
    with temporary.open("w", encoding="utf-8-sig", newline="") as file:
        for start in range(0, len(games), 5000):
            values = np.asarray(matrix[start:start + 5000])
            if not np.isfinite(values).all():
                raise ValueError(f"{combination}: nonfinite feature values.")
            stop = start + len(values)
            table = pd.DataFrame(values, columns=names)
            table.insert(0, "Name", games["Name"].iloc[start:stop].to_numpy())
            table.insert(0, "AppID", games["AppID"].iloc[start:stop].to_numpy())
            table.to_csv(file, index=False, header=(start == 0),
                         float_format="%.9g", lineterminator="\n")
            if start == 0:
                table.head(10).to_csv(preview_path, index=False, encoding="utf-8-sig",
                                      float_format="%.9g", lineterminator="\n")
            if start % 25000 == 0 or stop == len(games):
                print(f"{combination}: wrote {stop:,}/{len(games):,}", flush=True)

    # 3. Nine significant digits allow exact float32 values on reload.
    # Only replace the finished CSV after the read-back check passes.
    checked = check_csv(temporary, games, matrix, names)
    temporary.replace(csv_path)

    # 4. Keep the existing output format so old saved files remain compatible.
    result = {
        "combination": combination, "file": str(csv_path.resolve()),
        "preview_file": str(preview_path.resolve()),
        "rows": checked, "feature_columns": len(names), "total_columns": len(names) + 2,
        "bytes": csv_path.stat().st_size, "csv_sha256": file_sha256(csv_path),
        "source_feature_sha256": info["feature_sha256"],
        "every_coordinate_matches_float32_source": True,
        "format": "UTF-8 with BOM, comma-separated, nine significant decimal digits",
        "column_order": "AppID, Name, categorical block, description block if present, popularity if present",
    }
    write_json(folder / "embeddings_csv_manifest.json", result)
    print(f"{combination}: checked and saved {csv_path}", flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--combinations", nargs="+", choices=list("ABCD"), default=list("ABCD"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    root = args.output_dir.resolve()
    games = pd.read_csv(root / "game_index.csv", dtype={"AppID": np.int64, "Name": str},
                        keep_default_na=False)
    if games["AppID"].duplicated().any():
        parser.error("Duplicate AppIDs in the saved game index.")
    try:
        exports = []
        for combination in dict.fromkeys(args.combinations):
            exports.append(export_one(root, combination, games, args.overwrite))
        write_json(root / "embedding_csv_exports.json", {
            "exported_at_utc": datetime.now(timezone.utc).isoformat(),
            "exports": exports,
            "meaning": "Saved weighted embeddings used by K-means, not the original unnormalized source descriptors.",
        })
    except (ValueError, OSError) as error:
        parser.exit(2, f"CSV export failed: {error}\n")


if __name__ == "__main__":
    main()
