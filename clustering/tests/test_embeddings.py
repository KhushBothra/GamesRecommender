"""Description preparation checks using a local mock, with no model download."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from clustering import embeddings


class _Tokenizer:
    def num_special_tokens_to_add(self, pair=False):
        return 2

    def encode(self, text, **kwargs):
        return list(range(1, len(text.split()) + 1))

    def decode(self, tokens, **kwargs):
        return " ".join(f"w{token}" for token in tokens)


class _Encoder:
    builds = []
    calls = []

    def __init__(self, model_id, **kwargs):
        self.builds.append((model_id, kwargs))
        self.tokenizer = _Tokenizer()

    def encode(self, texts, **kwargs):
        self.calls.append(list(texts))
        return np.asarray([[len(text.split()), len(text), 1] for text in texts], dtype=np.float32)

    def __getitem__(self, index):
        return SimpleNamespace(auto_model=SimpleNamespace(config=SimpleNamespace(_commit_hash="mock-resolved-revision")))


class DescriptionEmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(
            prefix="game-embedding-test-", dir=Path(__file__).resolve().parent,
        )
        self.root = Path(self.temporary.name)
        self.arrays = []
        self.frame = {
            "AppID": [10, 20, 30],
            "Description": ["<p>Hello world</p>", "some game text", "alpha beta gamma delta epsilon"],
        }
        self.args = SimpleNamespace(
            cache_dir=self.root / "cache", description_embeddings=None,
            description_model="mock/model", model_revision="mock-pin",
            max_seq_length=4, chunk_descriptions=False, batch_size=2,
            device="cpu", offline=True, force=False, threads=2,
        )
        _Encoder.builds = []
        _Encoder.calls = []
        self.thread_counts = []
        self.modules = patch.dict(sys.modules, {
            "sentence_transformers": SimpleNamespace(SentenceTransformer=_Encoder),
            "torch": SimpleNamespace(set_num_threads=self.thread_counts.append),
        })
        self.modules.start()

    def tearDown(self):
        for array in self.arrays:
            embeddings._close_memmap(array)
        self.modules.stop()
        self.temporary.cleanup()

    def prepare(self, frame=None):
        result, metadata = embeddings.prepare_descriptions(self.frame if frame is None else frame, self.args)
        self.arrays.append(result)
        return result, metadata

    def external_fixture(self):
        array, metadata = self.prepare()
        path = self.root / "external.npy"
        np.save(path, np.asarray(array))
        metadata = {**metadata, "embedding_sha256": embeddings._sha256(path)}
        path.with_suffix(".json").write_text(json.dumps(metadata), encoding="utf-8")
        self.args.description_embeddings = path
        return path, metadata

    def test_cleaning_and_empty_inputs(self):
        self.assertEqual(embeddings.clean_description(
            "<p>Hello <b>world</b></p><script>bad()</script><style>.x{}</style> &amp; ok"
        ), "Hello world & ok")
        for value in (None, "", "   ", "<script>only code</script>"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare({"AppID": [1], "Description": [value]})
        self.assertEqual(len(_Encoder.builds), 0)

    def test_generated_cache_and_reuse(self):
        array, metadata = self.prepare()
        self.assertEqual(array.shape, (3, 3))
        self.assertEqual(array.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(array, axis=1), 1, rtol=1e-6)
        self.assertEqual(metadata["dimension"], 3)
        self.assertEqual(metadata["resolved_model_revision"], "mock-resolved-revision")
        self.assertEqual(metadata["torch_threads"], 2)
        self.assertEqual(self.thread_counts, [2])
        self.assertTrue(_Encoder.builds[0][1]["local_files_only"])
        again, repeated_metadata = self.prepare()
        np.testing.assert_array_equal(array, again)
        self.assertEqual(metadata, repeated_metadata)
        self.args.force = True
        self.prepare()
        self.assertEqual(len(_Encoder.builds), 1)
        self.assertFalse(list(self.args.cache_dir.glob("*.partial.npy")))

    def test_required_hashes_and_external_without_encoder(self):
        path, metadata = self.external_fixture()
        expected_ids = hashlib.sha256(np.asarray([10, 20, 30], dtype="<i8").tobytes()).hexdigest()
        text_digest = hashlib.sha256()
        for text in ("Hello world", "some game text", "alpha beta gamma delta epsilon"):
            encoded = text.encode("utf-8")
            text_digest.update(len(encoded).to_bytes(8, "little"))
            text_digest.update(encoded)
        self.assertEqual(metadata["app_ids_sha256"], expected_ids)
        self.assertEqual(metadata["description_text_sha256"], text_digest.hexdigest())
        with patch.dict(sys.modules, {"sentence_transformers": None, "torch": None}):
            array, _ = self.prepare()
        self.assertEqual(array.shape, (3, 3))
        self.assertEqual(len(_Encoder.builds), 1)
        changed = {"AppID": [20, 10, 30], "Description": self.frame["Description"]}
        with self.assertRaisesRegex(ValueError, "app_ids_sha256"):
            self.prepare(changed)
        changed = {"AppID": self.frame["AppID"], "Description": ["new text", "some game text", self.frame["Description"][2]]}
        with self.assertRaisesRegex(ValueError, "description_text_sha256"):
            self.prepare(changed)
        self.assertTrue(path.exists())

    def test_external_metadata_identity_fields(self):
        path, metadata = self.external_fixture()
        mutations = {
            "schema_version": 2, "model_id": "different/model", "model_revision": "different-pin",
            "max_seq_length": 5, "pooling": "mean_chunks", "rows": 2, "normalized": False,
            "dimension": 100,
        }
        for field, replacement in mutations.items():
            with self.subTest(field=field):
                path.with_suffix(".json").write_text(json.dumps({**metadata, field: replacement}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare()
        path.with_suffix(".json").write_text(json.dumps({key: value for key, value in metadata.items() if key != "model_revision"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "model_revision"):
            self.prepare()

    def test_external_bad_values_and_checksum(self):
        path, metadata = self.external_fixture()
        for value in (2.0, np.nan, np.inf):
            with self.subTest(value=value):
                altered = np.eye(3, dtype=np.float32)
                altered[0, 0] = value
                np.save(path, altered)
                path.with_suffix(".json").write_text(json.dumps({**metadata, "embedding_sha256": embeddings._sha256(path)}), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare()
        path.with_suffix(".json").write_text(json.dumps({**metadata, "embedding_sha256": "0" * 64}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            self.prepare()

    def test_chunked_mean_and_normalization(self):
        self.args.chunk_descriptions = True
        result, metadata = self.prepare({"AppID": [10], "Description": ["alpha beta gamma delta epsilon"]})
        expected = np.asarray([5 / 3, 4, 1], dtype=np.float64)
        expected /= np.linalg.norm(expected)
        np.testing.assert_allclose(result[0], expected, rtol=1e-6)
        self.assertEqual(metadata["pooling"], "mean_chunks")
        self.assertTrue(all(len(chunk.split()) <= 2 for batch in _Encoder.calls for chunk in batch))

    def test_incomplete_cache_needs_force(self):
        array, _ = self.prepare()
        embeddings._close_memmap(array)
        metadata_path = next(self.args.cache_dir.glob("descriptions_*.json"))
        metadata_path.unlink()
        with self.assertRaisesRegex(ValueError, "--force"):
            self.prepare()
        self.args.force = True
        self.prepare()
        self.assertEqual(len(_Encoder.builds), 2)


if __name__ == "__main__":
    unittest.main()
