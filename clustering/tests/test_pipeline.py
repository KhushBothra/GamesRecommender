"""Small, synthetic checks for the shared feature-preparation pipeline.

These tests need no Steam CSVs, model download, or trained clustering models.
Run from the project root with:
    python -m unittest discover -s clustering/tests -v
"""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from clustering.pipeline import (
    COMBINATIONS,
    combine_features,
    encode_popularity,
    load_inputs,
    normalize_rows,
)


POPULARITY_LABELS = [
    "Zero (0)",
    "Very low (1-10)",
    "Low (11-100)",
    "Moderate (101-1,000)",
    "High (1,001-10,000)",
    "Very high (10,001-100,000)",
    "Extreme (100,001+)",
]


class NormalizationTests(unittest.TestCase):
    def test_normalization_preserves_ratios_and_zero_rows(self):
        original = np.array([[3.0, 4.0], [0.0, 0.0], [0.0, 8.0]])
        result = normalize_rows(original)

        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result, [[0.6, 0.8], [0, 0], [0, 1]])
        np.testing.assert_array_equal(original, [[3, 4], [0, 0], [0, 8]])
        self.assertTrue(np.isfinite(result).all())

    def test_invalid_shapes_and_nonfinite_values_are_rejected(self):
        for values in (
            np.array([1, 2]),
            np.zeros((1, 1, 1)),
            np.array([[np.nan, 1.0]]),
            np.array([[np.inf, 1.0]]),
            np.array([[-np.inf, 1.0]]),
        ):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    normalize_rows(values)


class PopularityTests(unittest.TestCase):
    def test_all_existing_bins_use_ordered_codes(self):
        result = encode_popularity(POPULARITY_LABELS)
        self.assertEqual(result.shape, (7, 1))
        self.assertEqual(result.dtype, np.float32)
        np.testing.assert_allclose(result[:, 0], np.arange(7) / 6)

    def test_a_missing_bin_is_not_silently_treated_as_zero_popularity(self):
        np.testing.assert_array_equal(encode_popularity([POPULARITY_LABELS[0]]), [[0]])
        for invalid in (None, "", np.nan, "Unknown", "Zero", "low (11-100)"):
            with self.subTest(label=invalid):
                with self.assertRaises(ValueError):
                    encode_popularity([POPULARITY_LABELS[0], invalid])


class CombinationTests(unittest.TestCase):
    def setUp(self):
        self.categorical = normalize_rows(np.array([[1, 0.5, 0], [0, 0, 1]]))
        self.descriptions = normalize_rows(np.array([[1, -1], [0, 1]]))
        self.popularity = np.array([[0.0], [1.0]], dtype=np.float32)

    @staticmethod
    def weights(combination):
        entry = COMBINATIONS[combination]
        return entry.get("weights", entry)

    def test_combination_shapes_and_block_order(self):
        for combination, dimensions in (("A", 3), ("B", 4), ("C", 5), ("D", 6)):
            with self.subTest(combination=combination):
                result = combine_features(
                    combination, self.categorical, self.descriptions, self.popularity
                )
                self.assertEqual(result.shape, (2, dimensions))
                self.assertEqual(result.dtype, np.float32)
                weights = self.weights(combination)
                expected_blocks = [np.sqrt(weights["categorical"]) * self.categorical]
                if weights["description"]:
                    expected_blocks.append(np.sqrt(weights["description"]) * self.descriptions)
                if weights["popularity"]:
                    expected_blocks.append(np.sqrt(weights["popularity"]) * self.popularity)
                np.testing.assert_allclose(result, np.concatenate(expected_blocks, axis=1), atol=1e-7)

    def test_euclidean_distance_obeys_the_declared_block_weights(self):
        blocks = {
            "categorical": self.categorical,
            "description": self.descriptions,
            "popularity": self.popularity,
        }
        for combination in ("A", "B", "C", "D"):
            with self.subTest(combination=combination):
                result = combine_features(
                    combination, self.categorical, self.descriptions, self.popularity
                )
                weights = self.weights(combination)
                expected_distance_squared = sum(
                    weights[name] * float(np.sum((values[0] - values[1]) ** 2))
                    for name, values in blocks.items()
                )
                self.assertAlmostEqual(
                    float(np.sum((result[0] - result[1]) ** 2)),
                    expected_distance_squared,
                    places=6,
                )

    def test_popularity_information_survives_without_global_normalization(self):
        categorical = normalize_rows(np.array([[1, 0], [1, 0]]))
        descriptions = normalize_rows(np.array([[0, 1], [0, 1]]))
        result = combine_features("D", categorical, descriptions, self.popularity)

        # The final scalar must not rescale otherwise-identical content blocks.
        np.testing.assert_array_equal(result[0, :-1], result[1, :-1])
        self.assertAlmostEqual(float(result[0, -1]), 0.0)
        self.assertAlmostEqual(float(result[1, -1]), np.sqrt(0.1), places=6)
        self.assertNotAlmostEqual(float(np.linalg.norm(result[0])), float(np.linalg.norm(result[1])))

    def test_only_required_blocks_need_to_be_supplied(self):
        self.assertEqual(combine_features("A", self.categorical).shape, (2, 3))
        self.assertEqual(
            combine_features("B", self.categorical, popularity=self.popularity).shape,
            (2, 4),
        )
        self.assertEqual(
            combine_features("C", self.categorical, descriptions=self.descriptions).shape,
            (2, 5),
        )
        for combination in ("B", "C", "D"):
            with self.subTest(combination=combination):
                with self.assertRaises(ValueError):
                    combine_features(combination, self.categorical)

    def test_invalid_block_values_and_row_counts_are_rejected(self):
        cases = [
            (self.categorical, self.descriptions[:1], self.popularity),
            (self.categorical, self.descriptions, self.popularity[:1]),
            (self.categorical, self.descriptions, np.array([[-0.01], [1.0]])),
            (self.categorical, self.descriptions, np.array([[0.0], [1.01]])),
            (self.categorical, self.descriptions, np.array([[np.nan], [1.0]])),
            (np.array([[np.inf, 0], [1, 0]]), self.descriptions, self.popularity),
            (self.categorical, np.array([[np.nan, 0], [1, 0]]), self.popularity),
        ]
        for categorical, descriptions, popularity in cases:
            with self.subTest(categorical=categorical, descriptions=descriptions, popularity=popularity):
                with self.assertRaises(ValueError):
                    combine_features("D", categorical, descriptions, popularity)


class InputAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.cleaned_path = Path(self.directory.name) / "cleaned.csv"
        self.descriptor_path = Path(self.directory.name) / "descriptors.csv"
        self.cleaned = pd.DataFrame({
            "AppID": [20, 10, 30],
            "Name": ["Game twenty", "Game ten", "Game thirty"],
            "Description": ["Twenty description", "Ten description", "Thirty description"],
            "Peak CCU": [5, 0, 200],
            "Popularity Bin": [POPULARITY_LABELS[1], POPULARITY_LABELS[0], POPULARITY_LABELS[3]],
        })
        self.descriptors = pd.DataFrame({
            "AppID": [30, 20, 10],
            "Action": [0.0, 1.0, 0.5],
            "Single-player": [0.8, 0.4, 0.0],
        })

    def write_inputs(self, cleaned=None, descriptors=None):
        (self.cleaned if cleaned is None else cleaned).to_csv(self.cleaned_path, index=False)
        (self.descriptors if descriptors is None else descriptors).to_csv(self.descriptor_path, index=False)

    def test_shuffled_descriptors_are_joined_by_appid_not_position(self):
        self.write_inputs()
        result = load_inputs(self.cleaned_path, self.descriptor_path)

        self.assertEqual(result["frame"]["AppID"].tolist(), [20, 10, 30])
        self.assertEqual(result["descriptor_columns"], ["Action", "Single-player"])
        self.assertEqual(result["categorical"].dtype, np.float32)
        np.testing.assert_allclose(result["categorical"], [[1, 0.4], [0.5, 0], [0, 0.8]])
        self.assertIsInstance(result["source_signature"], dict)

    def test_duplicate_ids_in_either_input_are_rejected(self):
        for input_name in ("cleaned", "descriptors"):
            with self.subTest(input_name=input_name):
                cleaned = self.cleaned.copy()
                descriptors = self.descriptors.copy()
                target = cleaned if input_name == "cleaned" else descriptors
                target.loc[1, "AppID"] = target.loc[0, "AppID"]
                self.write_inputs(cleaned, descriptors)
                with self.assertRaises(ValueError):
                    load_inputs(self.cleaned_path, self.descriptor_path)

    def test_different_game_sets_are_rejected_before_sampling(self):
        descriptors = self.descriptors.copy()
        descriptors.loc[0, "AppID"] = 999
        self.write_inputs(descriptors=descriptors)
        for limit in (None, 1):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    load_inputs(self.cleaned_path, self.descriptor_path, limit=limit)

    def test_missing_required_columns_are_rejected(self):
        for missing in ("AppID", "Name", "Description", "Peak CCU", "Popularity Bin"):
            with self.subTest(missing=missing):
                self.write_inputs(cleaned=self.cleaned.drop(columns=missing))
                with self.assertRaises(ValueError):
                    load_inputs(self.cleaned_path, self.descriptor_path)
        self.write_inputs(descriptors=self.descriptors.drop(columns="AppID"))
        with self.assertRaises(ValueError):
            load_inputs(self.cleaned_path, self.descriptor_path)

    def test_invalid_descriptor_values_are_rejected(self):
        for invalid in (-0.01, 1.01, np.nan, np.inf, -np.inf, "not a number"):
            with self.subTest(value=invalid):
                descriptors = self.descriptors.astype({"Action": object})
                descriptors.loc[0, "Action"] = invalid
                self.write_inputs(descriptors=descriptors)
                with self.assertRaises(ValueError):
                    load_inputs(self.cleaned_path, self.descriptor_path)

    def test_malformed_csv_is_rejected(self):
        self.write_inputs()
        self.descriptor_path.write_text('AppID,Action\n30,"unterminated\n', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_inputs(self.cleaned_path, self.descriptor_path)


if __name__ == "__main__":
    unittest.main()
