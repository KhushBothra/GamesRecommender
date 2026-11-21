import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from read_data import load_inputs
"""Check that fixed-k averaging cannot silently become best-k averaging."""

import tempfile
import unittest
from pathlib import Path

from file_helpers import write_json
from make_comparison_table import make_table


class ComparisonTableTests(unittest.TestCase):
    def test_separate_k_seeds_missing_invalid_and_partial_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_json(root / 'experiment.json', {'smoke_test': True, 'settings': {
                'combinations': ['A', 'B'], 'clusters': [2, 3], 'seeds': [42, 43]}})

            def save(cell, k, status, silhouette, db, seed=42):
                score = dict(combination='A', region=cell, k=k, seed=seed,
                             status=status, silhouette=silhouette, davies_bouldin=db)
                path = root / 'A' / cell / f'k_{k:02d}_seed_{seed}' / 'result.json'
                write_json(path, dict(score=score, files={}, stability=[]))

            save('x1_y1', 2, 'ok', 0.2, 2.0)
            save('x1_y2', 2, 'ok', 0.4, 1.0)
            save('x1_y3', 2, 'invalid', None, None)
            save('x2_y1', 2, 'partial', None, 3.0)
            save('x1_y1', 3, 'ok', 0.9, 0.1)
            save('x1_y1', 2, 'ok', 0.8, 0.5, seed=43)

            table = make_table(root).set_index(['embedding', 'k', 'seed'])
            self.assertEqual(len(table), 8)
            row = table.loc[('A', 2, 42)]
            self.assertAlmostEqual(row.silhouette_mean, 0.3)
            self.assertAlmostEqual(row.db_mean, 2.0)
            self.assertEqual(row.silhouette_cells, 2)
            self.assertEqual(row.db_cells, 3)
            self.assertEqual(row.pending_cells, 5)
            self.assertEqual(row.invalid_cells, 1)
            self.assertAlmostEqual(row.valid_fraction, 2 / 9)
            self.assertAlmostEqual(table.loc[('A', 3, 42)].silhouette_mean, 0.9)
            self.assertAlmostEqual(table.loc[('A', 2, 43)].silhouette_mean, 0.8)
            self.assertEqual(table.loc[('B', 2, 42)].pending_cells, 9)
            self.assertEqual(table.loc[('B', 2, 42)].table_status, 'PROVISIONAL')


if __name__ == '__main__':
    unittest.main()
