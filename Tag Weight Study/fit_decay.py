"""Fit four positional decay families to verified SteamSpy empirical means.

Run `python fit_decay.py` from any directory. Import `weight_vector` to reuse a
fit without performing I/O. Input CSVs are never written by this module.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import minimize_scalar


FAMILIES = ("linear", "exponential", "logarithmic", "power")
PARAMETER_NAMES = dict(zip(FAMILIES, ("a", "lambda", "a", "p")))
EQUATIONS = {
    "linear": "w(r,k) = 1 - a*x",
    "exponential": "w(r,k) = exp(-lambda*x)",
    "logarithmic": "w(r,k) = 1 - a*ln(1+x)/ln(2)",
    "power": "w(r,k) = (1+x)^(-p)",
}
COLORS = dict(zip(FAMILIES, ("#0072B2", "#D55E00", "#009E73", "#CC79A7")))


@dataclass(frozen=True)
class Group:
    n_tags: int
    n_games: int
    weights: np.ndarray


def curve(x, family: str, parameter: float) -> np.ndarray:
    """Evaluate a constrained family on normalized positions in [0, 1]."""
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    x = np.asarray(x, dtype=float)
    if not np.all(np.isfinite(x)) or np.any((x < 0) | (x > 1)):
        raise ValueError("Positions must be finite and in [0, 1].")
    if not np.isfinite(parameter) or parameter < 0:
        raise ValueError("Parameter must be finite and nonnegative.")
    if family in ("linear", "logarithmic") and parameter > 1:
        raise ValueError("Linear/logarithmic parameter must be in [0, 1].")
    if family == "linear":
        return 1 - parameter * x
    if family == "logarithmic":
        return 1 - parameter * np.log1p(x) / np.log(2)
    basis = x if family == "exponential" else np.log1p(x)
    return np.exp(-parameter * basis)


def weight_vector(n_tags: int, family: str, parameter: float) -> np.ndarray:
    """Return ordered weights, including empty/singleton vectors.

    Lengths above the calibration maximum (21 for this dataset) are evaluated
    without clamping or truncation; these are extrapolations. No L2 scaling is
    applied here: rank 1 retains the study's max-normalized weight of 1.
    """
    if isinstance(n_tags, bool) or not isinstance(n_tags, Integral) or n_tags < 0:
        raise ValueError("n_tags must be a nonnegative integer.")
    curve(np.array([0.0]), family, parameter)  # Validate even empty inputs.
    if n_tags == 0:
        return np.empty(0)
    if n_tags == 1:
        return np.ones(1)
    return curve(np.linspace(0, 1, n_tags), family, parameter)


def load_targets(path: Path) -> list[Group]:
    """Validate targets; blanks outside each vector are never zero-filled."""
    groups = []
    seen = set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not {"n_tags", "n_games"} <= set(reader.fieldnames):
            raise ValueError("Targets need n_tags and n_games columns.")
        rank_columns = [c for c in reader.fieldnames if c.startswith("weight_rank_")]
        for row in reader:
            k, count = int(row["n_tags"]), int(row["n_games"])
            if k < 0 or count <= 0 or k in seen:
                raise ValueError(f"Invalid or duplicate group: {k}")
            seen.add(k)
            values = []
            for rank in range(1, k + 1):
                cell = row.get(f"weight_rank_{rank}")
                if cell is None or not cell.strip():
                    raise ValueError(f"Missing target at k={k}, rank={rank}")
                values.append(float(cell))
            for column in rank_columns:
                if int(column.removeprefix("weight_rank_")) > k and row[column].strip():
                    raise ValueError(f"Unexpected target beyond k={k}: {column}")
            y = np.array(values, dtype=float)
            if k and (not np.all(np.isfinite(y)) or np.any((y < 0) | (y > 1))
                      or not np.isclose(y[0], 1, rtol=0, atol=1e-12)
                      or np.any(np.diff(y) > 1e-12)):
                raise ValueError(f"Invalid normalized decay vector for k={k}")
            groups.append(Group(k, count, y))
    if not any(g.n_tags >= 2 for g in groups):
        raise ValueError("At least one group with two or more tags is required.")
    return sorted(groups, key=lambda g: g.n_tags)


def verify_inputs(study_dir: Path, groups: list[Group]) -> dict:
    """Rebuild every ranked source record and mean without modifying inputs."""
    source = study_dir / "Tag Weights DB/steamspy_tag_data.csv"
    indexed_path = study_dir / "steamspy_tag_data_indexed.csv"
    with indexed_path.open(encoding="utf-8-sig", newline="") as stream:
        indexed_rows = list(csv.DictReader(stream))
    indexed = {int(row["appid"]): row for row in indexed_rows}
    if len(indexed) != len(indexed_rows):
        raise ValueError("Duplicate AppIDs in indexed records.")
    sums, counts, seen = {}, {}, set()
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if not columns or "appid" not in columns or len(set(columns)) != len(columns):
            raise ValueError("Invalid or duplicate source column names.")
        tags = [c for c in columns if c != "appid"]
        for row in reader:
            appid = int(row["appid"])
            if appid in seen:
                raise ValueError(f"Duplicate source AppID: {appid}")
            seen.add(appid)
            values = [(tag, int(row[tag])) for tag in tags]
            if any(value < 0 for _, value in values):
                raise ValueError(f"Negative source count: {appid}")
            pairs = sorted(((t, v) for t, v in values if v > 0),
                           key=lambda pair: pair[1], reverse=True)
            k = len(pairs)
            ref = indexed.get(appid)
            if (ref is None or int(ref["n_tags"]) != k
                    or list(json.loads(ref["tag_counts"]).items()) != pairs):
                raise ValueError(f"Indexed record does not match source: {appid}")
            counts[k] = counts.get(k, 0) + 1
            if k:
                total = sums.setdefault(k, np.zeros(k))
                total += np.array([v / pairs[0][1] for _, v in pairs])
    if seen != set(indexed) or set(counts) != {g.n_tags for g in groups}:
        raise ValueError("Source, indexed records, and target groups differ.")
    max_difference = 0.0
    for group in groups:
        k = group.n_tags
        if counts[k] != group.n_games:
            raise ValueError(f"Group size mismatch for k={k}")
        if k:
            difference = float(np.max(np.abs(sums[k] / counts[k] - group.weights)))
            max_difference = max(max_difference, difference)
            if difference > 1e-12:
                raise ValueError(f"Empirical means do not match source for k={k}")
    paths = [source, indexed_path, study_dir / "steamspy_average_decay_by_n_tags.csv"]
    return {
        "source_rows": len(seen), "tag_columns": len(tags),
        "maximum_empirical_mean_difference": max_difference,
        "input_sha256": {str(p.relative_to(study_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in paths},
    }


def fit_parameter(groups: list[Group], family: str) -> float:
    """Minimize the mean of within-group MSEs, excluding rank 1."""
    informative = [g for g in groups if g.n_tags >= 2]
    if not informative:
        raise ValueError("No informative groups to fit.")
    if family not in FAMILIES:
        raise ValueError(f"Unknown family: {family}")
    x = np.concatenate([np.linspace(0, 1, g.n_tags)[1:] for g in informative])
    y = np.concatenate([g.weights[1:] for g in informative])
    weights = np.concatenate([np.full(g.n_tags - 1, 1 / (len(informative) * (g.n_tags - 1)))
                              for g in informative])

    def objective(parameter):
        return float(np.dot(weights, (curve(x, family, parameter) - y) ** 2))

    if family == "linear":
        return float(np.clip(np.dot(weights * x, 1 - y) / np.dot(weights, x * x), 0, 1))
    if family == "logarithmic":
        upper = 1.0
    else:
        basis = x if family == "exponential" else np.log1p(x)
        # Beyond the largest individual target crossing every squared-error
        # term is nondecreasing, so no smaller minimum lies outside this range.
        upper = float(np.max(-np.log(np.maximum(y, 1e-300)) / basis))
        if upper == 0:
            return 0.0
    # Scan linear and logarithmic grids before local refinement to avoid relying
    # on an assumed global unimodality of a sum of exponential residuals.
    grid = np.unique(np.concatenate(([0.0], np.linspace(0, upper, 257),
                                     np.geomspace(upper * 1e-10, upper, 257))))
    values = np.array([objective(p) for p in grid])
    candidates = [(float(values[0]), 0.0), (float(values[-1]), upper)]
    for i in range(1, len(grid) - 1):
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            result = minimize_scalar(objective, bounds=(grid[i - 1], grid[i + 1]),
                                     method="bounded", options={"xatol": 1e-12})
            if not result.success:
                raise RuntimeError(f"Optimization failed for {family}: {result.message}")
            candidates.append((float(result.fun), float(result.x)))
    return min(candidates)[1]


def aggregate_errors(rows: list[dict]) -> dict:
    """Aggregate per-group MSEs; RMSE is sqrt(aggregated MSE), not mean RMSE."""
    if not rows:
        raise ValueError("Cannot aggregate an empty set of groups.")
    macro_mse = math.fsum(row["mse"] for row in rows) / len(rows)
    n_games = sum(row["n_games"] for row in rows)
    weighted_mse = math.fsum(row["n_games"] * row["mse"] for row in rows) / n_games
    return {"group_balanced_mse": macro_mse, "group_balanced_rmse": math.sqrt(macro_mse),
            "sample_weighted_mse": weighted_mse, "sample_weighted_rmse": math.sqrt(weighted_mse)}


def compare(groups: list[Group]) -> dict:
    scores, parameters, errors, predictions = [], [], [], []
    informative = [g for g in groups if g.n_tags >= 2]
    if not informative:
        raise ValueError("No informative groups to compare.")
    scenarios = {"all_groups": informative}
    without_21 = [g for g in informative if g.n_tags != 21]
    if len(without_21) != len(informative) and without_21:
        scenarios["without_k21"] = without_21
    for scenario, selected in scenarios.items():
        for fit_type in ("shared", "per_length"):
            for family in FAMILIES:
                shared = fit_parameter(selected, family) if fit_type == "shared" else None
                model_errors = []
                if shared is not None:
                    parameters.append(dict(scenario=scenario, fit_type=fit_type, family=family,
                                           n_tags="", parameter_name=PARAMETER_NAMES[family],
                                           parameter=shared))
                for group in selected:
                    k = group.n_tags
                    parameter = shared if shared is not None else fit_parameter([group], family)
                    if shared is None:
                        parameters.append(dict(scenario=scenario, fit_type=fit_type, family=family,
                                               n_tags=k, parameter_name=PARAMETER_NAMES[family],
                                               parameter=parameter))
                    predicted = weight_vector(k, family, parameter)
                    mse = float(np.mean((predicted[1:] - group.weights[1:]) ** 2))
                    error = dict(scenario=scenario, fit_type=fit_type, family=family, n_tags=k,
                                 n_games=group.n_games, parameter=parameter, mse=mse, rmse=math.sqrt(mse))
                    model_errors.append(error)
                    errors.append(error)
                    for rank, (target, prediction) in enumerate(zip(group.weights, predicted), 1):
                        predictions.append(dict(scenario=scenario, fit_type=fit_type, family=family,
                                                n_tags=k, n_games=group.n_games, rank=rank,
                                                x=(rank - 1) / (k - 1), target=float(target),
                                                predicted=float(prediction), residual=float(prediction - target),
                                                included_in_score=rank > 1))
                scores.append(dict(scenario=scenario, fit_type=fit_type, family=family,
                                   n_groups=len(selected), n_games=sum(g.n_games for g in selected),
                                   n_parameters=1 if shared is not None else len(selected),
                                   **aggregate_errors(model_errors)))
    for scenario in scenarios:
        for fit_type in ("shared", "per_length"):
            ranked = sorted((s for s in scores if s["scenario"] == scenario and s["fit_type"] == fit_type),
                            key=lambda s: s["group_balanced_mse"])
            for rank, row in enumerate(ranked, 1):
                row["rank"] = rank
    return {"scores": scores, "parameters": parameters, "errors": errors, "predictions": predictions}


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output_dir: Path, groups: list[Group], result: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "savefig.facecolor": "white"})
    selected = [g for g in groups if g.n_tags >= 2]
    dense_x = np.linspace(0, 1, 301)
    for fit_type in ("shared", "per_length"):
        fig, axes = plt.subplots(math.ceil(len(selected) / 5), 5, figsize=(19, 12),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, group in zip(axes.flat, selected):
            ax.scatter(np.linspace(0, 1, group.n_tags), group.weights, c="#202A35", s=20,
                       label="Empirical mean", zorder=5)
            for family in FAMILIES:
                parameter = next(p["parameter"] for p in result["parameters"]
                                 if p["scenario"] == "all_groups" and p["fit_type"] == fit_type
                                 and p["family"] == family
                                 and (fit_type == "shared" or p["n_tags"] == group.n_tags))
                ax.plot(dense_x, curve(dense_x, family, parameter), color=COLORS[family],
                        lw=1.7, label=family.title())
            ax.set_title(f"k={group.n_tags} | n={group.n_games:,} games"
                         + (" (small sample)" if group.n_tags == 21 else ""), fontsize=11)
            ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04))
            ax.grid(alpha=0.15)
        for ax in list(axes.flat)[len(selected):]:
            ax.set_visible(False)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=5, frameon=False)
        title = "Shared parameters" if fit_type == "shared" else "Separate parameters for each list length"
        fig.suptitle(f"Steam tag decay: {title}", fontsize=20, y=0.995)
        fig.supxlabel("Normalized position x = (rank - 1) / (k - 1)", y=0.025)
        fig.supylabel("Mean max-normalized vote weight", x=0.008)
        fig.tight_layout(rect=(0.02, 0.04, 1, 0.935), h_pad=2)
        fig.savefig(output_dir / f"{fit_type}_fits.png", dpi=160)
        fig.savefig(output_dir / f"{fit_type}_fits.svg")
        plt.close(fig)
    scenarios = list(dict.fromkeys(s["scenario"] for s in result["scores"]))
    fig, axes = plt.subplots(1, len(scenarios), figsize=(7 * len(scenarios), 5), squeeze=False, sharey=True)
    for ax, scenario in zip(axes.flat, scenarios):
        for offset, fit_type, label, alpha in [(-0.19, "shared", "Shared (1 parameter)", 1),
                                                (0.19, "per_length", "Per length (1 per group)", 0.45)]:
            values = [next(s["group_balanced_rmse"] for s in result["scores"]
                           if s["scenario"] == scenario and s["fit_type"] == fit_type and s["family"] == f)
                      for f in FAMILIES]
            bars = ax.bar(np.arange(4) + offset, values, width=0.36, color=[COLORS[f] for f in FAMILIES],
                          alpha=alpha, label=label)
            ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.set_xticks(range(4), [f.title() for f in FAMILIES])
        ax.set_title("All groups (k=2–21)" if scenario == "all_groups" else "Sensitivity: exclude k=21")
        ax.set_ylabel("Group-balanced RMSE (lower is better)")
        ax.set_ylim(bottom=0)
        ax.margins(y=0.2)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Four decay families: shared rule versus separate fits", fontsize=17)
    fig.tight_layout()
    fig.savefig(output_dir / "model_comparison.png", dpi=160)
    fig.savefig(output_dir / "model_comparison.svg")
    plt.close(fig)


def select_model(result: dict) -> dict:
    scores = [s for s in result["scores"] if s["scenario"] == "all_groups" and s["fit_type"] == "shared"]
    winner = min(scores, key=lambda s: s["group_balanced_mse"])
    family = winner["family"]
    parameter = next(p["parameter"] for p in result["parameters"]
                     if p["scenario"] == "all_groups" and p["fit_type"] == "shared" and p["family"] == family)
    return {"family": family, "parameter_name": PARAMETER_NAMES[family], "parameter": parameter,
            "equation": EQUATIONS[family], "position": "x=(r-1)/(k-1) for k>1",
            "zero_tags": [], "one_tag": [1.0], "fit_type": "shared",
            "calibrated_n_tags_min": 2, "calibrated_n_tags_max": 21,
            "longer_lists": "Evaluate normalized position without truncation; extrapolation beyond calibration.",
            "selection_metric": "group_balanced_mse", "scores": winner}


def make_report(result: dict, model: dict, audit: dict) -> str:
    scores = result["scores"]
    full_shared = sorted((s for s in scores if s["scenario"] == "all_groups" and s["fit_type"] == "shared"),
                         key=lambda s: s["rank"])
    linear_parameter = next(p["parameter"] for p in result["parameters"]
                            if p["scenario"] == "all_groups" and p["fit_type"] == "shared" and p["family"] == "linear")
    lines = ["# Steam tag decay results", "",
             f"Selected shared curve: **{model['family']}**, `{model['equation']}`, "
             f"with `{model['parameter_name']}={model['parameter']:.10g}` and `x=(r-1)/(k-1)`.", "",
             f"The best straight line under the same weighting objective is **w = 1 - {linear_parameter:.10g} x**. "
             "Its intercept is fixed at 1; it is constrained to remain nonnegative.", "",
             "## Shared-model comparison", "",
             "| Rank | Family | Parameter | Group MSE | Group RMSE | Sample-weighted MSE | Sample-weighted RMSE |",
             "|---:|---|---:|---:|---:|---:|---:|"]
    for score in full_shared:
        parameter = next(p["parameter"] for p in result["parameters"]
                         if p["scenario"] == "all_groups" and p["fit_type"] == "shared" and p["family"] == score["family"])
        lines.append(f"| {score['rank']} | {score['family']} | {parameter:.9g} | "
                     f"{score['group_balanced_mse']:.9f} | {score['group_balanced_rmse']:.6f} | "
                     f"{score['sample_weighted_mse']:.9f} | {score['sample_weighted_rmse']:.6f} |")
    gap = (full_shared[1]["group_balanced_mse"] - full_shared[0]["group_balanced_mse"]) / full_shared[1]["group_balanced_mse"]
    weighted_winner = min(full_shared, key=lambda s: s["sample_weighted_mse"])["family"]
    lines += ["", f"The winner reduces group-balanced MSE by {100 * gap:.2f}% relative to the runner-up. "
              f"The sample-size-weighted winner is **{weighted_winner}**. "
              "Each shared family has one fitted parameter, so there is no parameter-count advantage between them.",
              "", "## Shared versus separate fits", "",
              "| Family | Shared RMSE | Per-length RMSE | MSE reduction |",
              "|---|---:|---:|---:|"]
    for shared in full_shared:
        separate = next(s for s in scores if s["scenario"] == "all_groups" and s["fit_type"] == "per_length"
                        and s["family"] == shared["family"])
        reduction = 1 - separate["group_balanced_mse"] / shared["group_balanced_mse"]
        lines.append(f"| {shared['family']} | {shared['group_balanced_rmse']:.6f} | "
                     f"{separate['group_balanced_rmse']:.6f} | {100 * reduction:.2f}% |")
    separate_winner = min((s for s in scores if s["scenario"] == "all_groups" and s["fit_type"] == "per_length"),
                          key=lambda s: s["group_balanced_mse"])["family"]
    lines += ["", f"The lowest-error per-length family is **{separate_winner}**. Separate fits use 20 parameters "
              "per family, compared with one for the shared rule. These are descriptive training fits; "
              "the lower error does not establish better generalization.", "",
              "A single normalized-position curve forces the same last-rank weight at every list length. "
              "The empirical last-rank means vary considerably with length, so the shared winner is a "
              "substantial compromise. Prefer the per-length results when the objective is descriptive "
              "accuracy for the observed lengths, and use the shared winner when one compact rule is required.",
              "", "## Sensitivity to the 21-tag group", ""]
    sensitivity = sorted((s for s in scores if s["scenario"] == "without_k21" and s["fit_type"] == "shared"),
                         key=lambda s: s["rank"])
    if sensitivity:
        changed = sensitivity[0]["family"] != model["family"]
        lines += ["The k=21 group contains only 12 games, and its last-rank mean is unusually low. "
                  "All shared models were refitted after excluding that group.", "",
                  "| Family | Group RMSE without k=21 | Sample-weighted RMSE without k=21 |",
                  "|---|---:|---:|"]
        lines += [f"| {s['family']} | {s['group_balanced_rmse']:.6f} | {s['sample_weighted_rmse']:.6f} |" for s in sensitivity]
        lines += ["", f"Winner without k=21: **{sensitivity[0]['family']}**. "
                  f"Winner changed: **{'yes' if changed else 'no'}**. "
                  "The full-data winner remains the saved selection; the group is not silently discarded."]
    else:
        lines += ["No 21-tag group was present for an exclusion comparison."]
    lines += ["", "## Metrics, verification, and limits", "",
              "For each k, MSE_k = mean((prediction - empirical_mean)^2) over ranks 2…k. "
              "Group-balanced MSE is the unweighted mean of MSE_k. Sample-weighted MSE is "
              "sum(N_k*MSE_k)/sum(N_k). Each RMSE is the square root of its corresponding aggregated MSE; "
              "it is not an average of per-group RMSEs. Shared fits minimize group-balanced MSE; "
              "sample weighting evaluates those same fitted parameters rather than refitting them.", "",
              f"Verified {audit['source_rows']:,} source records against the indexed CSV; "
              f"maximum recomputed empirical-mean difference: {audit['maximum_empirical_mean_difference']:.3g}. "
              "Zero-tag and singleton groups carry no decay information and are excluded from scores. "
              "Rank 1 is plotted but excluded from every score.", "",
              "This selects the closest curve among the four specified families, under normalized position "
              "and the stated metric. It does not establish recommendation quality, causal tag importance, "
              "or an optimal curve among all possible functions. No uncertainty intervals or held-out "
              "recommendation evaluation were performed. The tail anomaly in k=21 and variation across "
              "list lengths remain visible in the plots.", "",
              "Lists above 21 use the same formula at normalized positions and are extrapolations. "
              "The function does not clamp/truncate lists or apply L2 normalization. Genres, categories, "
              "and main preprocessing features are not changed.", "",
              "## Plots", "", "![Model comparison](model_comparison.png)", "",
              "![Shared fits by list length](shared_fits.png)", "",
              "![Separate fits by list length](per_length_fits.png)", "",
              "See model_scores.csv, fitted_parameters.csv, per_group_errors.csv, predictions.csv, "
              "empirical_targets.csv, selected_model.json, and run_manifest.json for reusable results.", ""]
    return "\n".join(lines)


def run_study(study_dir: Path | str = Path(__file__).resolve().parent,
              output_dir: Path | str | None = None) -> dict:
    study_dir = Path(study_dir).resolve()
    output_dir = Path(output_dir).resolve() if output_dir is not None else study_dir / "results"
    if output_dir == study_dir or output_dir in study_dir.parents:
        raise ValueError("Use a dedicated output directory, not the study directory or its ancestors.")
    groups = load_targets(study_dir / "steamspy_average_decay_by_n_tags.csv")
    audit = verify_inputs(study_dir, groups)
    result = compare(groups)
    model = select_model(result)
    model["calibrated_n_tags_min"] = min(g.n_tags for g in groups if g.n_tags >= 2)
    model["calibrated_n_tags_max"] = max(g.n_tags for g in groups if g.n_tags >= 2)
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, name in [("scores", "model_scores.csv"), ("parameters", "fitted_parameters.csv"),
                      ("errors", "per_group_errors.csv"), ("predictions", "predictions.csv")]:
        write_csv(output_dir / name, result[key])
    targets = [dict(n_tags=g.n_tags, n_games=g.n_games, rank=r, target=float(y))
               for g in groups for r, y in enumerate(g.weights, 1)]
    write_csv(output_dir / "empirical_targets.csv", targets)
    (output_dir / "selected_model.json").write_text(json.dumps(model, indent=2) + "\n", encoding="utf-8")
    manifest = {**audit, "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "matplotlib")},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fit_objective": "mean_k(mean_r((prediction-target)^2)); r=2..k; k>=2",
                "sensitivity": "refit after removing k=21; evaluate same parameters for sample weighting",
                "selection": "lowest full-data shared group-balanced MSE",
                "groups": [{"n_tags": g.n_tags, "n_games": g.n_games} for g in groups]}
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    make_plots(output_dir, groups, result)
    (output_dir / "RESULTS.md").write_text(make_report(result, model, audit), encoding="utf-8")
    return {**result, "selected_model": model, "audit": audit, "output_dir": output_dir}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    result = run_study(args.study_dir, args.output_dir)
    model = result["selected_model"]
    print(f"Selected {model['family']}: {model['parameter_name']}={model['parameter']:.10g}")
    print(f"Group-balanced RMSE: {model['scores']['group_balanced_rmse']:.8f}")
    print(f"Results: {result['output_dir'] / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
