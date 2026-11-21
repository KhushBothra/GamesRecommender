"""Exercise embedding creation, cache reuse, local clustering and tables together."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

import make_embeddings
import compare_embeddings
from make_comparison_table import save_tables
from feature_vectors import POPULARITY_LABELS


class SmallEncoder:
    loads = 0

    def __init__(self, *args, **kwargs):
        SmallEncoder.loads += 1

    def encode(self, texts, **kwargs):
        return np.array([[len(text), len(text.split()), 1] for text in texts], dtype=np.float32)


class FullRunTests(unittest.TestCase):
    def test_create_reuse_study_and_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cleaned, descriptors = root/'games.csv', root/'descriptors.csv'
            output, cache, results = root/'outputs', root/'cache', root/'results'
            pd.DataFrame({'AppID': range(10, 18), 'Name': [f'Game {i}' for i in range(8)],
                'Description': [f'Description {i} with ' + 'extra text ' * i for i in range(8)],
                'Peak CCU': range(8), 'Popularity Bin': [POPULARITY_LABELS[i % 7] for i in range(8)]}).to_csv(cleaned, index=False)
            terms = pd.DataFrame(np.random.default_rng(42).random((8, 3)), columns=['Action', 'Zombies', 'Co-op'])
            terms.insert(0, 'AppID', range(10, 18))
            terms.to_csv(descriptors, index=False)
            original = [cleaned.read_bytes(), descriptors.read_bytes()]
            arguments = ['--cleaned', str(cleaned), '--descriptors', str(descriptors),
                '--output-dir', str(output), '--cache-dir', str(cache), '--offline']
            modules = {'sentence_transformers': SimpleNamespace(SentenceTransformer=SmallEncoder),
                       'torch': SimpleNamespace(set_num_threads=lambda value: None)}
            SmallEncoder.loads = 0
            with patch.dict(sys.modules, modules):
                make_embeddings.main(arguments)
            stamps = {p: p.stat().st_mtime_ns for p in output.glob('*/features.npy')}
            with patch.dict(sys.modules, {'sentence_transformers': None, 'torch': None}):
                make_embeddings.main(arguments)
            self.assertEqual(SmallEncoder.loads, 1)
            self.assertTrue(all(p.stat().st_mtime_ns == stamp for p, stamp in stamps.items()))
            for name, dimension in [('A', 3), ('B', 4), ('C', 6), ('D', 7)]:
                self.assertEqual(np.load(output/name/'features.npy').shape, (8, dimension))

            study_args = ['--cleaned', str(cleaned), '--descriptors', str(descriptors),
                '--features-dir', str(output), '--output-dir', str(results), '--clusters', '2', '3',
                '--n-init', '1', '--resamples', '2', '--sample-size', '8', '--threads', '1']
            compare_embeddings.main(study_args)
            table = save_tables(results)
            self.assertEqual(len(table), 8)
            self.assertEqual(int(table.pending_cells.sum()), 0)
            self.assertEqual(len(list(results.glob('*/*/k_*/result.json'))), 72)
            self.assertEqual(json.loads((results/'experiment.json').read_text())['status'], 'complete')
            with patch('region_functions.KMeans', side_effect=AssertionError('Must not retrain')):
                compare_embeddings.main(study_args)
            self.assertEqual(original, [cleaned.read_bytes(), descriptors.read_bytes()])


if __name__ == '__main__':
    unittest.main()
