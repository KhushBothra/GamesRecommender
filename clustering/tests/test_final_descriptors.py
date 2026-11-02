import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from clustering.create_embeddings import export_or_reuse, load_official_inputs, main
from clustering.pipeline import file_sha256


class OfficialInputTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cleaned = self.root / "games_cleaned.csv"
        self.descriptors = self.root / "games_descriptors.csv"
        pd.DataFrame({
            "AppID": [10, 20], "Name": ["Alpha", "Beta"],
            "Description": ["A zombie game", "A farming game"],
            "Peak CCU": [0, 11], "Popularity Bin": ["Zero (0)", "Low (11-100)"],
        }).to_csv(self.cleaned, index=False)
        pd.DataFrame({
            "AppID": [20, 10], "Action": [0.125, 0.75], "Zombies": [0.25, 0.1],
        }).to_csv(self.descriptors, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_preweighted_values_preserved_and_sources_unchanged(self):
        before = [p.read_bytes() for p in (self.cleaned, self.descriptors)]
        data = load_official_inputs(self.cleaned, self.descriptors)
        np.testing.assert_array_equal(
            data["categorical"], np.array([[0.75, 0.1], [0.125, 0.25]], dtype=np.float32))
        self.assertEqual(data["frame"].AppID.tolist(), [10, 20])
        self.assertEqual(before, [p.read_bytes() for p in (self.cleaned, self.descriptors)])
        self.assertFalse((self.root / "games.csv").exists())

    def test_lfs_pointer_in_either_input_is_rejected(self):
        for path in (self.cleaned, self.descriptors):
            with self.subTest(path=path.name):
                original = path.read_bytes()
                path.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:" + "0"*64 + "\nsize 10\n")
                with self.assertRaisesRegex(ValueError, "LFS pointer"):
                    load_official_inputs(self.cleaned, self.descriptors)
                path.write_bytes(original)

    def test_missing_appid_is_not_silently_dropped(self):
        frame = pd.read_csv(self.descriptors)
        frame.loc[0, "AppID"] = 30
        frame.to_csv(self.descriptors, index=False)
        with self.assertRaisesRegex(ValueError, "different AppID"):
            load_official_inputs(self.cleaned, self.descriptors)

    def test_prepare_only_writes_nothing_and_does_not_encode(self):
        before = [p.read_bytes() for p in (self.cleaned, self.descriptors)]
        with patch("clustering.create_embeddings.prepare_descriptions", side_effect=AssertionError("Must not encode")):
            main(["--cleaned", str(self.cleaned), "--descriptors", str(self.descriptors),
                  "--output-dir", str(self.root/"outputs"), "--prepare-only"])
        self.assertFalse((self.root/"outputs").exists())
        self.assertEqual(before, [p.read_bytes() for p in (self.cleaned, self.descriptors)])


class CSVReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        directory = self.root/"A"
        directory.mkdir()
        self.path = directory/"embeddings.csv"
        self.path.write_text("AppID,Name,categorical__Action\n10,Alpha,0.75\n20,Beta,0.125\n")
        self.frame = pd.DataFrame({"AppID": [10, 20], "Name": ["Alpha", "Beta"]})
        self.features = {"feature_sha256": "f"*64, "dimension": 1}
        self.record = {
            "source_feature_sha256": "f"*64, "rows": 2, "feature_columns": 1,
            "every_coordinate_matches_float32_source": True, "csv_sha256": file_sha256(self.path),
        }
        (directory/"embeddings_csv_manifest.json").write_text(json.dumps(self.record))

    def tearDown(self):
        self.temp.cleanup()

    def test_verified_csv_is_reused_without_rewriting(self):
        before = self.path.stat().st_mtime_ns
        with patch("clustering.create_embeddings.export_one", side_effect=AssertionError("Must not export")):
            record = export_or_reuse(self.root, "A", self.frame, self.features)
        self.assertEqual(record, self.record)
        self.assertEqual(before, self.path.stat().st_mtime_ns)

    def test_changed_csv_is_rejected(self):
        self.path.write_text("corrupted CSV")
        with self.assertRaisesRegex(ValueError, "CSV failed validation"):
            export_or_reuse(self.root, "A", self.frame, self.features)


if __name__ == "__main__":
    unittest.main()
