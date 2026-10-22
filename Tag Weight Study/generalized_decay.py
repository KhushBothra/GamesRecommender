"""Train, select, and test a length-aware exponential decay on disjoint games.

The original four-family descriptive study is preserved. Outputs from this
experiment live under results/generalization. Selection uses validation games;
reported test errors use training-only fits. A separate all-data refit is saved
for reuse after the evaluation.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
from numbers import Integral
from pathlib import Path
import platform

import numpy as np
from scipy.optimize import least_squares

from fit_decay import (FAMILIES, Group, aggregate_errors, fit_parameter, load_targets,
                       verify_inputs, weight_vector, write_csv)


DEFAULT_SEED = 20261023
SMOOTH_MODELS = ("length_affine_exponential", "length_power_exponential",
                 "length_saturating_exponential")
RATE_EQUATIONS = {
    "raw_rank_exponential": "lambda(k) = beta*(k-1)",
    "length_affine_exponential": "lambda(k) = a + b*(k-1)",
    "length_power_exponential": "lambda(k) = a*(k-1)^b",
    "length_saturating_exponential": "lambda(k) = a*(1-exp(-b*(k-1)))",
}
LABELS = {**{f"shared_{f}": f"Shared {f}" for f in FAMILIES},
          "per_length_exponential": "Per-length exponential",
          "raw_rank_exponential": "Absolute-rank exponential",
          "length_affine_exponential": "Length-aware: affine rate",
          "length_power_exponential": "Length-aware: power rate",
          "length_saturating_exponential": "Length-aware: saturating rate"}


@dataclass(frozen=True)
class Games:
    appids: np.ndarray
    weights: np.ndarray


def load_games(path: Path) -> dict[int, Games]:
    """Read positive descending vote counts and max-normalize each game."""
    rows, seen = {}, set()
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            appid, k = int(row["appid"]), int(row["n_tags"])
            if appid in seen or k < 0:
                raise ValueError("Duplicate AppID or negative tag-list length.")
            seen.add(appid)
            counts = np.asarray(list(json.loads(row["tag_counts"]).values()), dtype=float)
            if (len(counts) != k or not np.all(np.isfinite(counts)) or np.any(counts <= 0)
                    or np.any(counts != np.floor(counts)) or np.any(np.diff(counts) > 0)):
                raise ValueError(f"Invalid ranked counts for AppID {appid}")
            vector = counts / counts[0] if k else np.empty(0)
            rows.setdefault(k, []).append((appid, vector))
    result = {}
    for k, values in sorted(rows.items()):
        values.sort(key=lambda item: item[0])
        result[k] = Games(np.array([v[0] for v in values], dtype=np.int64),
                          np.array([v[1] for v in values], dtype=float))
    return result


def split_games(games: dict[int, Games], seed: int = DEFAULT_SEED) -> tuple[dict, list[dict]]:
    """Deterministic 60/20/20 game split within each length, sorted by AppID."""
    rng = np.random.default_rng(seed)
    splits = {s: {} for s in ("train", "validation", "test")}
    assignments = []
    for k, block in sorted(games.items()):
        if k < 2:
            assignments.extend(dict(appid=int(appid), n_tags=k, split="excluded_no_decay") for appid in block.appids)
            continue
        n = len(block.appids)
        n_train, n_validation = math.floor(0.6 * n), math.floor(0.2 * n)
        if min(n_train, n_validation, n - n_train - n_validation) < 1:
            raise ValueError(f"Length {k} has too few games for a three-way split.")
        order = rng.permutation(n)
        index_sets = np.split(order, [n_train, n_train + n_validation])
        for split, indices in zip(splits, index_sets):
            indices.sort()
            splits[split][k] = Games(block.appids[indices], block.weights[indices])
            assignments.extend(dict(appid=int(appid), n_tags=k, split=split) for appid in block.appids[indices])
    if not splits["train"]:
        raise ValueError("No informative games to split.")
    return splits, sorted(assignments, key=lambda row: row["appid"])


def group_means(games: dict[int, Games]) -> list[Group]:
    return [Group(k, len(block.appids), block.weights.mean(axis=0))
            for k, block in sorted(games.items()) if k >= 2]


def rate(n_tags, model: dict):
    """Evaluate a nonnegative length-dependent rate (k >= 2)."""
    k = np.asarray(n_tags, dtype=float)
    if not np.all(np.isfinite(k)) or np.any(k < 2) or np.any(k != np.floor(k)):
        raise ValueError("Rate inputs must be integer lengths >= 2.")
    name, params = model["name"], model["parameters"]
    if name not in RATE_EQUATIONS:
        raise ValueError(f"Not a length-rate model: {name}")
    if any(not np.isfinite(p) or p < 0 for p in params.values()):
        raise ValueError("Rate parameters must be finite and nonnegative.")
    d = k - 1
    if name == "raw_rank_exponential":
        return params["beta"] * d
    a, b = params["a"], params["b"]
    if name == "length_affine_exponential":
        return a + b * d
    if name == "length_power_exponential":
        if b > 1:
            raise ValueError("The power-rate exponent must be in [0, 1].")
        return a * d ** b
    return a * (-np.expm1(-b * d))


def predict_weights(n_tags: int, model: dict) -> np.ndarray:
    """Public weight function for saved models; lengths >21 are extrapolations.

    Per-length lookup models raise ValueError at an uncalibrated length. Smooth
    models extend their documented equation, with no clipping/truncation.
    """
    if isinstance(n_tags, bool) or not isinstance(n_tags, Integral) or n_tags < 0:
        raise ValueError("n_tags must be a nonnegative integer.")
    name = model["name"]
    valid = {f"shared_{f}" for f in FAMILIES} | set(RATE_EQUATIONS) | {"per_length_exponential"}
    if name not in valid:
        raise ValueError(f"Unknown model: {name}")
    if n_tags < 2:
        return np.ones(n_tags)
    if name.startswith("shared_"):
        return weight_vector(n_tags, name.removeprefix("shared_"), model["parameters"]["parameter"])
    if name == "per_length_exponential":
        try:
            parameter = model["parameters"][str(n_tags)]
        except KeyError as error:
            raise ValueError(f"No independently fitted rate for length {n_tags}.") from error
    else:
        parameter = float(rate(n_tags, model))
    return weight_vector(n_tags, "exponential", parameter)


def fit_rate_model(groups: list[Group], name: str) -> dict:
    """Jointly fit mean curves, with each length contributing equal total weight."""
    groups = [g for g in groups if g.n_tags >= 2]
    if not groups or name not in RATE_EQUATIONS:
        raise ValueError("Need informative groups and a supported rate model.")
    lengths = np.concatenate([np.full(g.n_tags - 1, g.n_tags) for g in groups])
    positions = np.concatenate([np.linspace(0, 1, g.n_tags)[1:] for g in groups])
    target = np.concatenate([g.weights[1:] for g in groups])
    scales = np.concatenate([np.full(g.n_tags - 1, 1 / math.sqrt(len(groups) * (g.n_tags - 1))) for g in groups])
    raw = name == "raw_rank_exponential"
    names = ("beta",) if raw else ("a", "b")
    starts = ([(0.05,), (0.15,), (0.3,)] if raw else
              [(0.1, 0.1), (0.5, 0.05), (1, 0.5)] if name == "length_affine_exponential" else
              [(0.2, 0.7), (0.5, 0.5), (1, 0.1), (0.1, 0.95)] if name == "length_power_exponential" else
              [(2, 0.1), (3, 0.05), (1, 0.5), (10, 0.01)])
    upper = [np.inf] if raw else [np.inf, 1 if name == "length_power_exponential" else np.inf]

    def residual(parameters):
        model = {"name": name, "parameters": dict(zip(names, parameters))}
        return (np.exp(-rate(lengths, model) * positions) - target) * scales

    solutions = []
    for start in starts:
        solution = least_squares(residual, start, bounds=(np.zeros(len(names)), upper),
                                 ftol=1e-12, xtol=1e-12, gtol=1e-12, max_nfev=5000)
        if solution.success and np.all(np.isfinite(solution.x)):
            solutions.append(solution)
    if not solutions:
        raise RuntimeError(f"No successful optimization for {name}")
    best = min(solutions, key=lambda solution: np.dot(solution.fun, solution.fun))
    return {"name": name, "parameters": {key: float(value) for key, value in zip(names, best.x)},
            "n_parameters": len(names), "rate_equation": RATE_EQUATIONS[name],
            "weight_equation": "w(r,k) = exp(-lambda(k)*(r-1)/(k-1))"}


def fit_candidates(groups: list[Group]) -> list[dict]:
    models = [{"name": f"shared_{family}", "parameters": {"parameter": fit_parameter(groups, family)},
               "n_parameters": 1} for family in FAMILIES]
    models.append({"name": "per_length_exponential", "n_parameters": len(groups),
                   "parameters": {str(g.n_tags): fit_parameter([g], "exponential") for g in groups}})
    models.extend(fit_rate_model(groups, name) for name in ("raw_rank_exponential", *SMOOTH_MODELS))
    return models


def evaluate_model(model: dict, games: dict[int, Games], split: str) -> tuple[list[dict], dict]:
    errors, predictions = [], []
    for k, block in sorted(games.items()):
        if k < 2:
            continue
        predicted = predict_weights(k, model)
        mean = block.weights.mean(axis=0)
        mean_mse = float(np.mean((predicted[1:] - mean[1:]) ** 2))
        game_mse = float(np.mean((block.weights[:, 1:] - predicted[None, 1:]) ** 2))
        variance = float(np.mean((block.weights[:, 1:] - mean[None, 1:]) ** 2))
        errors.append(dict(model=model["name"], split=split, n_tags=k, n_games=len(block.appids),
                           mean_curve_mse=mean_mse, mean_curve_rmse=math.sqrt(mean_mse),
                           game_mse=game_mse, game_rmse=math.sqrt(game_mse), within_group_variance=variance))
        for rank, (target, prediction) in enumerate(zip(mean, predicted), 1):
            predictions.append(dict(model=model["name"], split=split, n_tags=k, n_games=len(block.appids),
                                    rank=rank, target=float(target), predicted=float(prediction),
                                    included_in_score=rank > 1))
    summaries = []
    for scenario in ("all_lengths", "without_k21"):
        selected = [r for r in errors if scenario == "all_lengths" or r["n_tags"] != 21]
        if not selected:
            continue
        summary = dict(model=model["name"], split=split, scenario=scenario,
                       n_parameters=model["n_parameters"], n_groups=len(selected),
                       n_games=sum(r["n_games"] for r in selected))
        for prefix, field in (("mean_curve", "mean_curve_mse"), ("game", "game_mse")):
            aggregated = aggregate_errors([{"mse": r[field], "n_games": r["n_games"]} for r in selected])
            summary.update({f"{prefix}_{key}": value for key, value in aggregated.items()})
        summaries.append(summary)
    return summaries, {"errors": errors, "predictions": predictions}


def evaluate_experiment(splits: dict) -> dict:
    """Select on validation, then score test; no test inputs enter fitting."""
    models = fit_candidates(group_means(splits["train"]))
    scores, errors, predictions = [], [], []
    # Selection happens before the test dictionary is read by evaluation.
    for split in ("train", "validation"):
        for model in models:
            summary, detail = evaluate_model(model, splits[split], split)
            scores.extend(summary)
            errors.extend(detail["errors"])
            predictions.extend(detail["predictions"])
    validation = [s for s in scores if s["split"] == "validation" and s["scenario"] == "all_lengths"]
    metric = "mean_curve_group_balanced_mse"
    smooth_choice = min((s for s in validation if s["model"] in SMOOTH_MODELS), key=lambda s: s[metric])["model"]
    overall_choice = min(validation, key=lambda s: s[metric])["model"]
    for model in models:
        summary, detail = evaluate_model(model, splits["test"], "test")
        scores.extend(summary)
        errors.extend(detail["errors"])
        predictions.extend(detail["predictions"])
    return {"models": models, "scores": scores, "errors": errors, "predictions": predictions,
            "selected_smooth": smooth_choice, "selected_overall": overall_choice}


def paired_bootstrap(games: dict[int, Games], smooth: dict, baseline: dict,
                     seed: int, replicates: int = 2000) -> dict:
    """Conditional test-set uncertainty for smooth-minus-baseline MSE.

    For fixed predictions, the difference of mean-curve MSEs equals the
    difference of mean individual-game MSEs: within-group variance cancels.
    Thus bootstrapping per-game squared-error differences covers both metrics.
    """
    rng = np.random.default_rng(seed)
    deltas, bootstraps = {}, {}
    for k, block in sorted(games.items()):
        p, q = predict_weights(k, smooth)[1:], predict_weights(k, baseline)[1:]
        difference = np.mean((block.weights[:, 1:] - p) ** 2 - (block.weights[:, 1:] - q) ** 2, axis=1)
        deltas[k] = float(difference.mean())
        indices = rng.integers(0, len(difference), size=(replicates, len(difference)))
        bootstraps[k] = difference[indices].mean(axis=1)
    results = {}
    for scenario in ("all_lengths", "without_k21"):
        keys = [k for k in deltas if scenario == "all_lengths" or k != 21]
        if not keys:
            continue
        sampled = np.array([bootstraps[k] for k in keys]).mean(axis=0)
        low, high = np.quantile(sampled, [0.025, 0.975])
        results[scenario] = {"mse_difference": float(np.mean([deltas[k] for k in keys])),
                             "ci95_lower": float(low), "ci95_upper": float(high),
                             "interval_includes_zero": bool(low <= 0 <= high)}
    return {"comparison": f"{smooth['name']} minus {baseline['name']}", "replicates": replicates,
            "seed": seed, "positive_means": "smooth model has higher MSE",
            "method": "paired bootstrap of test games within each length; equal length weighting",
            "limits": "Conditional on the fitted models; excludes training/selection uncertainty and new datasets.",
            "results": results}


def leave_length_out(train: dict[int, Games], test: dict[int, Games], name: str) -> list[dict]:
    """Stress-test the validation-selected form without training on each length."""
    rows = []
    groups = group_means(train)
    if len(groups) < 2:
        raise ValueError("Leave-length-out needs at least two training lengths.")
    for held in sorted(train):
        retained = [g for g in groups if g.n_tags != held]
        model = fit_rate_model(retained, name)
        _, detail = evaluate_model(model, {held: test[held]}, "test")
        row = detail["errors"][0]
        row["held_out_length"] = held
        row["length_extrapolation"] = held < min(g.n_tags for g in retained) or held > max(g.n_tags for g in retained)
        row["parameters"] = json.dumps(model["parameters"], sort_keys=True)
        rows.append(row)
    return rows


def make_plots(output: Path, result: dict, splits: dict, final_model: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = result["selected_smooth"]
    testing = sorted((s for s in result["scores"] if s["split"] == "test" and s["scenario"] == "all_lengths"),
                     key=lambda s: s["mean_curve_group_balanced_mse"])
    fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True)
    colors = ["#00856A" if s["model"] == selected else "#D67621" if s["model"] == "per_length_exponential" else "#617D98"
              for s in testing]
    for ax, field, title in zip(axes, ("mean_curve_group_balanced_rmse", "game_group_balanced_rmse"),
                                ("Held-out group means", "Held-out individual games")):
        values = [s[field] for s in testing]
        bars = ax.barh(np.arange(len(testing)), values, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=4)
        ax.set(xlabel="Group-balanced test RMSE (lower is better)", title=title, xlim=(0, max(values) * 1.18))
        ax.set_yticks(np.arange(len(testing)), [LABELS[s["model"]] for s in testing])
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.suptitle("Generalization to held-out games | rate form selected on validation only", fontsize=16)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"test_comparison.{suffix}", dpi=160)
    plt.close(fig)
    smooth = next(m for m in result["models"] if m["name"] == selected)
    separate = next(m for m in result["models"] if m["name"] == "per_length_exponential")
    ks = np.array(sorted(splits["train"]))
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.scatter(ks, [separate["parameters"][str(k)] for k in ks], color="#D67621", label="Separate training fits", zorder=4)
    ax.plot(ks, rate(ks, smooth), color="#00856A", lw=2.5, label="Smooth fit: training only (evaluated)")
    ax.plot(ks, rate(ks, final_model), color="#617D98", lw=2, ls="--", label="Smooth fit: all data (for reuse)")
    ax.set(xlabel="Number of tags k", ylabel="Exponential rate lambda(k)",
           title=f"{LABELS[selected]} | {smooth['rate_equation']}")
    ax.set_xticks(ks)
    ax.legend(frameon=False)
    ax.grid(alpha=0.15)
    fig.tight_layout()
    for suffix in ("png", "svg"):
        fig.savefig(output / f"length_rates.{suffix}", dpi=160)
    plt.close(fig)
    blocks = sorted(splits["test"].items())
    fig, axes = plt.subplots(math.ceil(len(blocks) / 5), 5, figsize=(19, 12), sharex=True, sharey=True, squeeze=False)
    for ax, (k, block) in zip(axes.flat, blocks):
        x = np.linspace(0, 1, k)
        ax.scatter(x, block.weights.mean(axis=0), s=22, color="#202A35", label="Test mean", zorder=4)
        for model, color, label in [(smooth, "#00856A", "Smooth, training only"),
                                     (separate, "#D67621", "Per length, training only")]:
            parameter = float(rate(k, model)) if model["name"] != "per_length_exponential" else model["parameters"][str(k)]
            dense = np.linspace(0, 1, 301)
            ax.plot(dense, np.exp(-parameter * dense), color=color, lw=2, label=label)
        ax.set_title(f"k={k} | test n={len(block.appids):,}" + (" (sparse)" if k == 21 else ""), fontsize=11)
        ax.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.04))
        ax.grid(alpha=0.15)
    for ax in list(axes.flat)[len(blocks):]:
        ax.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.965), ncol=3, frameon=False)
    fig.suptitle("Test means versus exponential curves fitted on training games", fontsize=19, y=0.995)
    fig.supxlabel("Normalized position x = (rank - 1) / (k - 1)", y=0.025)
    fig.supylabel("Max-normalized vote weight", x=0.008)
    fig.tight_layout(rect=(0.02, 0.04, 1, 0.935), h_pad=2)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"test_curves.{suffix}", dpi=160)
    plt.close(fig)


def make_report(result: dict, splits: dict, final_model: dict, bootstrap: dict, lolo: list[dict], seed: int) -> str:
    selected = result["selected_smooth"]
    smooth = next(m for m in result["models"] if m["name"] == selected)
    lines = ["# Length-aware exponential decay: generalization experiment", "",
             f"Validation-selected smooth model: **{LABELS[selected]}**. `{smooth['rate_equation']}`; "
             "`w(r,k)=exp(-lambda(k)*(r-1)/(k-1))`.", "",
             f"Training-only evaluation parameters: `{json.dumps(smooth['parameters'], sort_keys=True)}`.", "",
             f"All-data parameters for reuse: `{json.dumps(final_model['parameters'], sort_keys=True)}`. "
             "The reported holdout scores do not evaluate this all-data refit.", "",
             "## Protocol", "",
             f"Seed **{seed}**. Sort by AppID, then randomly split games independently within every length "
             "into floor(60%*N) training, floor(20%*N) validation, and the remainder test. "
             "All tags/ranks from one game stay together. Zero-tag and singleton games are excluded from decay scores.", "",
             "| Split | Games | Lengths | Games at k=21 |", "|---|---:|---:|---:|"]
    for split, blocks in splits.items():
        lines.append(f"| {split} | {sum(len(b.appids) for b in blocks.values()):,} | {len(blocks)} | "
                     f"{len(blocks[21].appids) if 21 in blocks else 0} |")
    lines += ["", "Normalize each game's positive vote counts by that game's first count. Fit models to "
              "training means only, minimizing the unweighted mean of within-length MSEs over ranks 2…k. "
              "Validation selects the smooth rate form by held-out mean-curve group-balanced MSE. "
              "All nine candidates are frozen before test scoring; neither test means nor full-data means "
              "are passed to their fits. The best validation candidate across all models is "
              f"**{LABELS[result['selected_overall']]}**. Test results do not change that selection.", "",
              "Three smooth candidates have two nonnegative parameters: affine `a+b*(k-1)`, "
              "power `a*(k-1)^b` with `0<=b<=1`, and saturating `a*(1-exp(-b*(k-1)))`. "
              "The one-parameter absolute-rank baseline has `lambda(k)=beta*(k-1)`, equivalently "
              "`w=exp(-beta*(r-1))`. Shared original families each have one parameter; independent "
              "exponential fits have one parameter per observed length.", "", "## Validation comparison", "",
              "| Model | Parameters | Mean-curve macro RMSE | Individual-game macro RMSE |",
              "|---|---:|---:|---:|"]
    for split in ("validation", "test"):
        if split == "test":
            lines += ["", "## Held-out test comparison", "",
                      "| Model | Parameters | Mean-curve macro RMSE | Individual-game macro RMSE |",
                      "|---|---:|---:|---:|"]
        rows = sorted((s for s in result["scores"] if s["split"] == split and s["scenario"] == "all_lengths"),
                      key=lambda s: s["mean_curve_group_balanced_mse"])
        for row in rows:
            mark = " **(selected smooth)**" if row["model"] == selected else ""
            lines.append(f"| {LABELS[row['model']]}{mark} | {row['n_parameters']} | "
                         f"{row['mean_curve_group_balanced_rmse']:.6f} | {row['game_group_balanced_rmse']:.6f} |")
    test_scores = {s["model"]: s for s in result["scores"] if s["split"] == "test" and s["scenario"] == "all_lengths"}
    a, b = test_scores[selected], test_scores["per_length_exponential"]
    difference = a["mean_curve_group_balanced_mse"] - b["mean_curve_group_balanced_mse"]
    lines += ["", f"Selected smooth minus per-length exponential test mean-curve MSE: **{difference:+.8f}**. "
              "Positive values favor separate rates; negative values favor the smooth rate. "
              "Compare the absolute errors as well as the parameter counts when choosing the practical rule.", "",
              "## Test uncertainty and sparse-length sensitivity", "",
              "| Test scoring | Smooth RMSE | Per-length RMSE | Paired MSE difference | Conditional 95% interval |",
              "|---|---:|---:|---:|---|"]
    for scenario, interval in bootstrap["results"].items():
        by_name = {s["model"]: s for s in result["scores"] if s["split"] == "test" and s["scenario"] == scenario}
        lines.append(f"| {scenario} | {by_name[selected]['mean_curve_group_balanced_rmse']:.6f} | "
                     f"{by_name['per_length_exponential']['mean_curve_group_balanced_rmse']:.6f} | "
                     f"{interval['mse_difference']:+.8f} | [{interval['ci95_lower']:+.8f}, {interval['ci95_upper']:+.8f}] |")
    lines += ["", f"Paired bootstrap: {bootstrap['replicates']:,} resamples of test games within each length, "
              "with fixed fitted curves. Intervals cover test-sampling uncertainty only, conditional on "
              "training and validation choices. Rescoring without k=21 uses the same fitted models; "
              "it neither refits models nor changes the selected rate form. Only three k=21 games are in test, "
              "so its mean and uncertainty remain poorly estimated.", "", "## Generalization to omitted lengths", ""]
    lolo_summary = aggregate_errors([{"mse": r["mean_curve_mse"], "n_games": r["n_games"]} for r in lolo])
    lines += [f"With the validation-selected rate form fixed, omit each length from training, refit the "
              f"two parameters on the other lengths, and predict that length's test games. Mean-curve "
              f"group-balanced RMSE: **{lolo_summary['group_balanced_rmse']:.6f}**. "
              "The edge lengths are extrapolations; interior lengths are interpolations. "
              "This is a stress test of the chosen form, not nested model selection for wholly unseen lengths. "
              "Per-length lookup models cannot directly predict an omitted length.", "", "## Interpretation and limits", "",
              "Mean-curve MSE averages squared error against each held-out group's empirical mean. "
              "Individual-game MSE averages squared error against each game's normalized vector. "
              "Individual-game error = mean-curve error + within-group variation; all components "
              "are exported per length. Macro metrics give each length equal weight; sample-weighted "
              "metrics weight group MSE by the number of held-out games. Every RMSE is the square root "
              "of its aggregated MSE. Rank 1 is excluded throughout.", "",
              "This is a retrospective holdout within the historical dataset already explored in the "
              "descriptive study. It measures prediction of unseen games under this split, not a new "
              "time period, a new dataset, or recommendation quality. Small validation/test groups "
              "make family rankings uncertain. All-data refit parameters are for reuse and have no "
              "independent test score here. Lengths above the observed maximum are extrapolations.", "",
              "## Reuse", "", "```python", "import json", "from pathlib import Path",
              "from generalized_decay import predict_weights", "",
              "model = json.loads(Path('results/generalization/selected_length_aware_model.json').read_text())",
              "weights = predict_weights(10, model)", "```", "",
              "Zero and one tags return `[]` and `[1]`. The function keeps max normalization, "
              "does not truncate longer lists, and does not change main preprocessing features.", "",
              "## Plots", "", "![Test comparison](test_comparison.png)", "",
              "![Rates by length](length_rates.png)", "", "![Curves and test means](test_curves.png)", ""]
    return "\n".join(lines)


def run_generalization(study_dir: Path | str = Path(__file__).resolve().parent,
                       output_dir: Path | str | None = None, seed: int = DEFAULT_SEED) -> dict:
    root = Path(study_dir).resolve()
    output = Path(output_dir).resolve() if output_dir is not None else root / "results/generalization"
    if output == root or output in root.parents:
        raise ValueError("Use a dedicated output directory.")
    groups = load_targets(root / "steamspy_average_decay_by_n_tags.csv")
    audit = verify_inputs(root, groups)
    games = load_games(root / "steamspy_tag_data_indexed.csv")
    splits, assignments = split_games(games, seed)
    result = evaluate_experiment(splits)
    smooth = next(m for m in result["models"] if m["name"] == result["selected_smooth"])
    separate = next(m for m in result["models"] if m["name"] == "per_length_exponential")
    bootstrap = paired_bootstrap(splits["test"], smooth, separate, seed + 1)
    lolo = leave_length_out(splits["train"], splits["test"], smooth["name"])
    final_model = fit_rate_model(group_means(games), smooth["name"])
    final_model.update({"fit_data": "all games; refitted after frozen holdout evaluation",
                        "selected_using": "validation mean-curve group-balanced MSE",
                        "calibrated_n_tags_min": min(splits["train"]),
                        "calibrated_n_tags_max": max(splits["train"]),
                        "longer_lists": "Evaluate the rate formula without truncation; extrapolation.",
                        "zero_tags": [], "one_tag": [1.0]})
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in [("split_assignments.csv", assignments), ("model_scores.csv", result["scores"]),
                       ("per_group_errors.csv", result["errors"]), ("predictions.csv", result["predictions"]),
                       ("leave_length_out_errors.csv", lolo)]:
        write_csv(output / name, rows)
    rates = [dict(n_tags=k, train_games=len(splits["train"][k].appids),
                  validation_games=len(splits["validation"][k].appids), test_games=len(splits["test"][k].appids),
                  separate_training_lambda=separate["parameters"][str(k)],
                  smooth_training_lambda=float(rate(k, smooth)), smooth_all_data_lambda=float(rate(k, final_model)))
             for k in sorted(splits["train"])]
    write_csv(output / "length_rates.csv", rates)
    split_means = [dict(split=split, n_tags=k, n_games=len(block.appids), rank=r, target=float(value))
                   for split, blocks in splits.items() for k, block in sorted(blocks.items())
                   for r, value in enumerate(block.weights.mean(axis=0), 1)]
    write_csv(output / "split_targets.csv", split_means)
    artifacts = {"training_models.json": result["models"], "selected_evaluation_model.json": smooth,
                 "selected_length_aware_model.json": final_model, "paired_bootstrap.json": bootstrap}
    for name, value in artifacts.items():
        (output / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    manifest = {**audit, "seed": seed, "split": "within-length train=floor(0.6*N), validation=floor(0.2*N), test=remainder",
                "rng": "numpy.default_rng/PCG64; groups sorted by length; rows sorted by AppID",
                "selection_metric": "validation mean_curve_group_balanced_mse",
                "selected_smooth": result["selected_smooth"], "selected_overall": result["selected_overall"],
                "test_used_for_fitting_or_selection": False,
                "all_data_refit_separate_from_evaluated_models": True,
                "python": platform.python_version(),
                "packages": {p: importlib.metadata.version(p) for p in ("numpy", "scipy", "pandas", "matplotlib")},
                "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "fitting_module_sha256": hashlib.sha256((Path(__file__).parent / "fit_decay.py").read_bytes()).hexdigest()}
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    make_plots(output, result, splits, final_model)
    (output / "RESULTS.md").write_text(make_report(result, splits, final_model, bootstrap, lolo, seed), encoding="utf-8")
    return {**result, "final_model": final_model, "bootstrap": bootstrap, "leave_length_out": lolo,
            "output_dir": output, "splits": splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    result = run_generalization(args.study_dir, args.output_dir, args.seed)
    print("Selected smooth model:", result["selected_smooth"])
    print("All-data refit parameters:", result["final_model"]["parameters"])
    for row in result["scores"]:
        if row["split"] == "test" and row["scenario"] == "all_lengths":
            print(f"{row['model']}: mean-curve RMSE={row['mean_curve_group_balanced_rmse']:.6f}; "
                  f"game RMSE={row['game_group_balanced_rmse']:.6f}")
    print("Results:", result["output_dir"] / "RESULTS.md")


if __name__ == "__main__":
    main()
