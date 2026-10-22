"""Check leakage isolation, parameter recovery, and holdout error accounting."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fit_decay import Group
from generalized_decay import (Games, SMOOTH_MODELS, evaluate_experiment, evaluate_model,
                               fit_rate_model, group_means, leave_length_out, load_games,
                               paired_bootstrap, predict_weights, rate, split_games)


def synthetic_games():
    rng = np.random.default_rng(48)
    model = {"name": "length_saturating_exponential", "parameters": {"a": 2.2, "b": 0.1}}
    games = {}
    for k in (2, 4, 7, 12):
        vectors = np.clip(predict_weights(k, model) + rng.normal(0, 0.02, (30, k)), 0, 1)
        vectors[:, 0] = 1
        vectors = np.minimum.accumulate(vectors, axis=1)
        games[k] = Games(np.arange(k * 100, k * 100 + 30), vectors)
    return games


class GeneralizationTests(unittest.TestCase):
    def test_splits_are_disjoint_stratified_and_reproducible(self):
        games = synthetic_games()
        games[0] = Games(np.array([1]), np.empty((1, 0)))
        games[1] = Games(np.array([2]), np.ones((1, 1)))
        splits, assignments = split_games(games, 123)
        again, same = split_games(games, 123)
        self.assertEqual(assignments, same)
        self.assertEqual(len(assignments), sum(len(g.appids) for g in games.values()))
        self.assertEqual([a["split"] for a in assignments[:2]], ["excluded_no_decay"] * 2)
        for k in (2, 4, 7, 12):
            self.assertEqual([len(splits[s][k].appids) for s in splits], [18, 6, 6])
            sets = [set(splits[s][k].appids) for s in splits]
            self.assertEqual(set.union(*sets), set(games[k].appids))
            self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
            for split in splits:
                np.testing.assert_array_equal(splits[split][k].appids, again[split][k].appids)

    def test_constraints_and_unseen_lengths(self):
        for name in SMOOTH_MODELS:
            for a, b in ((0, 0), (2.2, 0.1), (1, 1)):
                model = {"name": name, "parameters": {"a": a, "b": b}}
                rates = rate(np.arange(2, 101), model)
                self.assertTrue(np.all(rates >= 0))
                self.assertTrue(np.all(np.diff(rates) >= -1e-12))
                for k in (0, 1, 2, 21, 40, 100):
                    vector = predict_weights(k, model)
                    self.assertEqual(len(vector), k)
                    if k:
                        self.assertEqual(vector[0], 1)
                    self.assertTrue(np.all((vector >= 0) & (vector <= 1)))
                    self.assertTrue(np.all(np.diff(vector) <= 0))
        with self.assertRaisesRegex(ValueError, "No independently fitted rate"):
            predict_weights(40, {"name": "per_length_exponential", "parameters": {"2": 0.3}})
        with self.assertRaises(ValueError):
            predict_weights(-1, model)
        with self.assertRaises(ValueError):
            rate(10, {"name": "length_power_exponential", "parameters": {"a": 1, "b": 1.1}})

    def test_synthetic_rate_parameter_recovery(self):
        specifications = {"raw_rank_exponential": {"beta": 0.14},
                          "length_affine_exponential": {"a": 0.12, "b": 0.09},
                          "length_power_exponential": {"a": 0.3, "b": 0.65},
                          "length_saturating_exponential": {"a": 2.25, "b": 0.095}}
        for name, parameters in specifications.items():
            expected = {"name": name, "parameters": parameters}
            groups = [Group(k, 100, predict_weights(k, expected)) for k in (2, 3, 6, 10, 15, 21)]
            fitted = fit_rate_model(groups, name)
            with self.subTest(model=name):
                for key, value in parameters.items():
                    self.assertAlmostEqual(fitted["parameters"][key], value, places=6)

    def test_absolute_rank_formula(self):
        model = {"name": "raw_rank_exponential", "parameters": {"beta": 0.14}}
        for k in (2, 5, 20, 40):
            np.testing.assert_allclose(predict_weights(k, model), np.exp(-0.14 * np.arange(k)))

    def test_error_decomposition_and_rank_one_exclusion(self):
        games = synthetic_games()
        model = {"name": "length_saturating_exponential", "parameters": {"a": 2.2, "b": 0.1}, "n_parameters": 2}
        summary, detail = evaluate_model(model, games, "test")
        for row in detail["errors"]:
            self.assertAlmostEqual(row["game_mse"], row["mean_curve_mse"] + row["within_group_variance"], places=14)
            k = row["n_tags"]
            expected = np.mean((games[k].weights[:, 1:] - predict_weights(k, model)[1:]) ** 2)
            self.assertAlmostEqual(row["game_mse"], expected)
        full = summary[0]
        self.assertAlmostEqual(full["mean_curve_group_balanced_mse"],
                               np.mean([r["mean_curve_mse"] for r in detail["errors"]]))
        self.assertAlmostEqual(full["game_group_balanced_rmse"] ** 2, full["game_group_balanced_mse"])
        self.assertTrue(all(p["included_in_score"] == (p["rank"] > 1) for p in detail["predictions"]))

    def test_test_games_do_not_affect_fits_or_model_selection(self):
        splits, _ = split_games(synthetic_games(), 10)
        original = evaluate_experiment(splits)
        changed_test = {k: Games(block.appids, np.ones_like(block.weights)) for k, block in splits["test"].items()}
        changed = evaluate_experiment({**splits, "test": changed_test})
        self.assertEqual(original["models"], changed["models"])
        self.assertEqual(original["selected_smooth"], changed["selected_smooth"])
        self.assertEqual(original["selected_overall"], changed["selected_overall"])
        self.assertEqual([s for s in original["scores"] if s["split"] != "test"],
                         [s for s in changed["scores"] if s["split"] != "test"])
        self.assertNotEqual([s for s in original["scores"] if s["split"] == "test"],
                            [s for s in changed["scores"] if s["split"] == "test"])

    def test_bootstrap_difference_matches_both_error_metrics(self):
        games = synthetic_games()
        smooth = {"name": "length_saturating_exponential", "parameters": {"a": 2.2, "b": 0.1}, "n_parameters": 2}
        baseline = {"name": "per_length_exponential", "parameters": {str(k): 0.8 for k in games}, "n_parameters": 4}
        result = paired_bootstrap(games, smooth, baseline, 12, replicates=200)
        again = paired_bootstrap(games, smooth, baseline, 12, replicates=200)
        self.assertEqual(result, again)
        a = evaluate_model(smooth, games, "test")[0][0]
        b = evaluate_model(baseline, games, "test")[0][0]
        delta = result["results"]["all_lengths"]["mse_difference"]
        self.assertAlmostEqual(delta, a["mean_curve_group_balanced_mse"] - b["mean_curve_group_balanced_mse"])
        self.assertAlmostEqual(delta, a["game_group_balanced_mse"] - b["game_group_balanced_mse"])

    def test_omitted_length_does_not_influence_its_fit(self):
        splits, _ = split_games(synthetic_games(), 10)
        name = "length_saturating_exponential"
        original = leave_length_out(splits["train"], splits["test"], name)
        train = dict(splits["train"])
        train[4] = Games(train[4].appids, np.ones_like(train[4].weights))
        changed = leave_length_out(train, splits["test"], name)
        before = next(r for r in original if r["held_out_length"] == 4)
        after = next(r for r in changed if r["held_out_length"] == 4)
        self.assertEqual(before, after)
        self.assertFalse(before["length_extrapolation"])
        self.assertTrue(next(r for r in original if r["held_out_length"] == 2)["length_extrapolation"])

    def test_loading_checks_vote_order_and_normalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "games.csv"
            path.write_text('appid,n_tags,tag_counts\n1,2,"{""a"":10,""b"":5}"\n')
            np.testing.assert_array_equal(load_games(path)[2].weights, [[1, 0.5]])
            path.write_text(path.read_text().replace('""a"":10,""b"":5', '""a"":5,""b"":10'))
            with self.assertRaisesRegex(ValueError, "Invalid ranked counts"):
                load_games(path)


if __name__ == "__main__":
    unittest.main()
