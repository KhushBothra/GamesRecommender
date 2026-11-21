import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from read_data import load_inputs
"""Check the scientific steps with small, known examples."""

import unittest

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from region_functions import make_lens, make_regions, score_groups, check_stability
from compare_embeddings import summarize


class RegionTests(unittest.TestCase):
    def test_lens_keeps_full_vectors_unchanged(self):
        vectors = np.array([[-2., 0], [-1, 0], [0, 0], [1, 0], [2, 0]])
        original = vectors.copy()
        lens, mean, direction, details = make_lens(vectors, percentile=50)
        np.testing.assert_allclose(mean, [0, 0])
        np.testing.assert_allclose(direction, [1, 0], atol=1e-6)
        np.testing.assert_allclose(lens[:, 0], [2, 1, 0, 1, 2])
        np.testing.assert_allclose(lens[:, 2], [-1, -1, 0, 1, 1])
        self.assertEqual(details['exact_mean_games'], 1)
        self.assertEqual(details['radius_clipped_games'], 2)
        np.testing.assert_array_equal(vectors, original)

    def test_identical_games_have_finite_lens(self):
        lens, _, _, details = make_lens(np.ones((8, 3)))
        np.testing.assert_array_equal(lens, np.zeros((8, 3)))
        self.assertEqual(details['exact_mean_games'], 8)

    def test_cover_keeps_overlaps_and_covers_endpoints(self):
        lens = np.array([[0, 0, -1], [1, 1, 1], [0.4, 0.4, -0.2]])
        regions = make_regions(lens)
        memberships = [r['indices'].tolist() for r in regions]
        self.assertEqual(len(regions), 9)
        self.assertEqual(set(i for cell in memberships for i in cell), {0, 1, 2})
        self.assertGreater(sum(2 in cell for cell in memberships), 1)

    def test_scores_use_original_dimensions_and_detect_invalid_sample(self):
        vectors = np.array([[0, 0, 0], [0, 0, 1], [0, 0, 10], [0, 0, 11.]])
        labels = np.array([0, 0, 1, 1])
        scores = score_groups(vectors, labels, np.arange(4))
        self.assertAlmostEqual(scores['silhouette'], silhouette_score(vectors, labels))
        self.assertEqual(scores['status'], 'ok')
        partial = score_groups(vectors, labels, np.array([0, 1]))
        self.assertEqual(partial['status'], 'partial')
        self.assertIsNone(partial['silhouette'])
        self.assertIsNotNone(partial['davies_bouldin'])

    def test_summary_includes_invalid_regions(self):
        rows = pd.DataFrame([
            dict(combination='A', k=2, seed=42, status='ok', silhouette=0.8, davies_bouldin=0.2),
            dict(combination='A', k=2, seed=42, status='invalid', silhouette=None, davies_bouldin=None)])
        row = summarize(rows).iloc[0]
        self.assertEqual(row.valid_fraction, 0.5)
        self.assertEqual(row.silhouette_local_macro_mean, 0.8)

    def test_stability_compares_shared_games(self):
        vectors = np.vstack([np.arange(10)[:, None], 100 + np.arange(10)[:, None]])
        pairs = check_stability(vectors, 2, 42, 3, 0.8, 2)
        self.assertEqual(len(pairs), 3)
        self.assertTrue(all(pair['ari'] == 1 for pair in pairs))
        self.assertTrue(all(0 < pair['shared_games'] <= 16 for pair in pairs))


if __name__ == '__main__':
    unittest.main()
