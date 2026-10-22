"""Scientific checks for fitting, scoring, and preservation of empirical inputs."""

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fit_decay import (FAMILIES, Group, aggregate_errors, compare, curve,
                       fit_parameter, load_targets, select_model, verify_inputs,
                       weight_vector)


class DecayTests(unittest.TestCase):
    def test_constraints_and_extrapolation(self):
        for family in FAMILIES:
            parameters = (0, 0.6, 1) if family in ("linear", "logarithmic") else (0, 0.6, 4, 1000)
            for parameter in parameters:
                for k in (0, 1, 2, 21, 40):
                    with self.subTest(family=family, parameter=parameter, k=k):
                        result = weight_vector(k, family, parameter)
                        self.assertEqual(len(result), k)
                        if k:
                            self.assertEqual(result[0], 1)
                        self.assertTrue(np.all((result >= 0) & (result <= 1)))
                        self.assertTrue(np.all(np.diff(result) <= 0))
        self.assertEqual(weight_vector(40, "linear", 0.6)[-1], 0.4)

    def test_invalid_public_inputs(self):
        for k in (-1, 2.5, True):
            with self.assertRaises(ValueError):
                weight_vector(k, "linear", 0.5)
        for family, parameter in (("unknown", 1), ("linear", 1.1), ("power", -1),
                                  ("exponential", float("nan")), ("logarithmic", float("inf"))):
            with self.assertRaises(ValueError):
                weight_vector(0, family, parameter)

    def test_shared_and_individual_parameter_recovery(self):
        for family in FAMILIES:
            expected = 0.63 if family in ("linear", "logarithmic") else 1.73
            groups = [Group(k, k * 100, weight_vector(k, family, expected)) for k in (2, 5, 12, 21)]
            with self.subTest(family=family):
                self.assertAlmostEqual(fit_parameter(groups, family), expected, places=6)
                for group in groups:
                    self.assertAlmostEqual(fit_parameter([group], family), expected, places=6)

    def test_fits_at_parameter_boundaries(self):
        for family in FAMILIES:
            for expected in (0, 1):
                with self.subTest(family=family, parameter=expected):
                    group = Group(10, 100, weight_vector(10, family, expected))
                    self.assertAlmostEqual(fit_parameter([group], family), expected, places=6)

    def test_group_balance_and_anchored_linear_solution(self):
        groups = [Group(2, 100000, np.array([1.0, 0.8])),
                  Group(3, 1, np.array([1.0, 0.8, 0.2]))]
        # Group-balanced least squares: numerator=.2 + (.5*.2+1*.8)/2;
        # denominator=1 + (.5**2+1**2)/2. Sample sizes do not enter the fit.
        expected = (0.2 + (0.5 * 0.2 + 0.8) / 2) / (1 + (0.25 + 1) / 2)
        self.assertAlmostEqual(fit_parameter(groups, "linear"), expected)
        swapped = [Group(g.n_tags, 1 if i == 0 else 100000, g.weights) for i, g in enumerate(groups)]
        self.assertEqual(fit_parameter(groups, "linear"), fit_parameter(swapped, "linear"))

    def test_error_aggregation_uses_mse_then_sqrt(self):
        result = aggregate_errors([{"mse": 0.01, "n_games": 1}, {"mse": 0.09, "n_games": 9}])
        self.assertAlmostEqual(result["group_balanced_mse"], 0.05)
        self.assertAlmostEqual(result["sample_weighted_mse"], 0.082)
        self.assertAlmostEqual(result["group_balanced_rmse"], np.sqrt(0.05))
        self.assertNotAlmostEqual(result["group_balanced_rmse"], (0.1 + 0.3) / 2)

    def test_blank_cells_are_not_zero_targets(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "targets.csv"
            path.write_text("n_tags,n_games,weight_rank_1,weight_rank_2,weight_rank_3\n"
                            "0,2,,,\n1,4,1,,\n2,10,1,0.6,\n3,20,1,0.8,0.4\n")
            groups = load_targets(path)
            self.assertEqual([len(g.weights) for g in groups], [0, 1, 2, 3])
            parameter = fit_parameter(groups, "linear")
            self.assertAlmostEqual(parameter, fit_parameter(groups[2:], "linear"))
            path.write_text(path.read_text().replace("2,10,1,0.6,", "2,10,1,,"))
            with self.assertRaisesRegex(ValueError, "Missing target"):
                load_targets(path)

    def test_rejects_invalid_empirical_vectors(self):
        for row in ("2,10,0.9,0.5", "2,10,1,1.1", "2,10,1,nan", "2,10,1,-0.2"):
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "targets.csv"
                path.write_text("n_tags,n_games,weight_rank_1,weight_rank_2\n" + row + "\n")
                with self.assertRaises(ValueError):
                    load_targets(path)

    def test_actual_inputs_reproduce_and_selection_is_consistent(self):
        root = Path(__file__).resolve().parent
        groups = load_targets(root / "steamspy_average_decay_by_n_tags.csv")
        audit = verify_inputs(root, groups)
        self.assertEqual(audit["source_rows"], 29022)
        self.assertEqual(audit["maximum_empirical_mean_difference"], 0)
        result = compare(groups)
        self.assertEqual(len(result["scores"]), 16)
        self.assertEqual(len(result["errors"]), (20 + 19) * 8)
        model = select_model(result)
        scores = [s for s in result["scores"] if s["scenario"] == "all_groups" and s["fit_type"] == "shared"]
        self.assertEqual(model["family"], min(scores, key=lambda s: s["group_balanced_mse"])["family"])
        # Recompute every table score from its underlying exported residuals.
        for score in result["scores"]:
            rows = [r for r in result["errors"] if all(r[k] == score[k] for k in ("scenario", "fit_type", "family"))]
            for row in rows:
                predictions = [p for p in result["predictions"] if p["n_tags"] == row["n_tags"]
                               and all(p[k] == score[k] for k in ("scenario", "fit_type", "family"))
                               and p["included_in_score"]]
                self.assertEqual(len(predictions), row["n_tags"] - 1)
                self.assertAlmostEqual(row["mse"], np.mean([p["residual"] ** 2 for p in predictions]))
            summary = aggregate_errors(rows)
            for name, value in summary.items():
                self.assertAlmostEqual(score[name], value)
        for family in FAMILIES:
            for scenario in ("all_groups", "without_k21"):
                matching = {s["fit_type"]: s for s in result["scores"]
                            if s["scenario"] == scenario and s["family"] == family}
                self.assertLessEqual(matching["per_length"]["group_balanced_mse"],
                                     matching["shared"]["group_balanced_mse"] + 1e-12)
        # Sensitivity models are actually refitted, not just rescored.
        for family in FAMILIES:
            parameter = next(p["parameter"] for p in result["parameters"]
                             if p["scenario"] == "without_k21" and p["fit_type"] == "shared" and p["family"] == family)
            self.assertEqual(parameter, fit_parameter([g for g in groups if g.n_tags != 21], family))

    def test_audit_rejects_stale_index(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "Tag Weights DB").mkdir()
            (root / "Tag Weights DB/steamspy_tag_data.csv").write_text("appid,a,b\n1,10,5\n")
            with (root / "steamspy_tag_data_indexed.csv").open("w", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["appid", "n_tags", "tag_counts"])
                writer.writerow([1, 2, json.dumps({"a": 10, "b": 4})])
            with self.assertRaisesRegex(ValueError, "does not match source"):
                verify_inputs(root, [Group(2, 1, np.array([1.0, 0.5]))])


if __name__ == "__main__":
    unittest.main()
