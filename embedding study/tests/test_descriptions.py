"""Check the supported description workflow without downloading a model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import numpy as np
import description_vectors as descriptions


class DescriptionTests(unittest.TestCase):
    def test_html_cleaning_and_rejected_empty_text(self):
        text = '<p>Hello <b>world</b></p><script>bad()</script><style>.x{}</style> &amp; ok'
        self.assertEqual(descriptions.clean_description(text), 'Hello world & ok')
        for value in [None, '', ' ', '<script>hidden</script>']:
            with self.assertRaises(ValueError):
                descriptions.prepare_text({'AppID': [1], 'Description': [value]})

    def test_cache_identity_and_corruption(self):
        class Encoder:
            def __init__(self, *args, **kwargs):
                pass
            def encode(self, texts, **kwargs):
                return np.array([[len(text), 1, 2] for text in texts], dtype=np.float32)

        with tempfile.TemporaryDirectory() as folder:
            args = SimpleNamespace(cache_dir=Path(folder), batch_size=2, threads=1, offline=True)
            frame = {'AppID': [1, 2], 'Description': ['first game', 'second game']}
            modules = {'sentence_transformers': SimpleNamespace(SentenceTransformer=Encoder),
                       'torch': SimpleNamespace(set_num_threads=lambda value: None)}
            with patch.dict(sys.modules, modules):
                first, info = descriptions.prepare_descriptions(frame, args)
            with patch.dict(sys.modules, {'sentence_transformers': None, 'torch': None}):
                second, _ = descriptions.prepare_descriptions(frame, args)
            np.testing.assert_array_equal(first, second)
            np.testing.assert_allclose(np.linalg.norm(second, axis=1), 1, atol=1e-6)
            path = next(Path(folder).glob('*.npy'))
            expected = {key: info[key] for key in ['app_ids_sha256', 'description_text_sha256', 'rows']}
            with self.assertRaises(ValueError):
                descriptions.load_cache(path, {**expected, 'app_ids_sha256': 'wrong'})
            path.write_bytes(b'bad cache')
            with self.assertRaises(ValueError):
                descriptions.load_cache(path, expected)


if __name__ == '__main__':
    unittest.main()
