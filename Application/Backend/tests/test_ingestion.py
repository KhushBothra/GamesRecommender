from __future__ import annotations

import csv
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from qdrant_client import QdrantClient, models

from Backend.infra.qdrant_connector import QdrantConnectorError
from Backend.tasks.ingest_dataset import ingest


class IngestionTests(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.games = Path(directory.name) / "games.csv"
        self.descriptors = Path(directory.name) / "descriptors.csv"
        self.client = QdrantClient(":memory:")
        self.addCleanup(self.client.close)
        self.write_games()
        self.write_descriptors()
        self.check = self.enterContext(patch.object(ingest, "check_qdrant_connection"))
        self.get_client = self.enterContext(patch.object(ingest, "get_qdrant_client", return_value=self.client))
        self.enterContext(patch.object(ingest, "settings", SimpleNamespace(qdrant_collection_name="test_games")))
        self.enterContext(redirect_stdout(io.StringIO()))

    def write_games(self, ids=(10, 20), price="10.0", headers=None):
        columns = list(headers or ingest.PAYLOAD_COLUMNS)
        with self.games.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(columns)
            for appid in ids:
                payload = dict(zip(ingest.PAYLOAD_COLUMNS, [appid, f"Game {appid}", "A game description.", "100", "Popular", '["Developer"]', '["Publisher"]', price]))
                writer.writerow([payload.get(column, "") for column in columns])

    def write_descriptors(self, rows=None, headers=None):
        with self.descriptors.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(headers or ["AppID", "Adventure", "Exploration", "Single-player"])
            writer.writerows(rows if rows is not None else [[10, 0.5, 1, 1], [20, 1, 0, 1]])

    def run_ingest(self, **kwargs):
        return ingest.ingest_dataset(self.games, self.descriptors, **kwargs)

    def create_collection(self, size=3, distance=models.Distance.COSINE, named=False):
        config = models.VectorParams(size=size, distance=distance)
        self.client.create_collection("test_games", vectors_config={"named": config} if named else config)

    def test_validation_only_does_not_contact_qdrant(self):
        self.assertEqual(self.run_ingest(validate_only=True), 2)
        self.check.assert_not_called()
        self.get_client.assert_not_called()

    def test_missing_collection_is_created_only_with_explicit_flag(self):
        with self.assertRaisesRegex(QdrantConnectorError, "--create-collection"):
            self.run_ingest()
        self.assertFalse(self.client.collection_exists("test_games"))
        self.assertEqual(self.run_ingest(create_collection=True, batch_size=1), 2)
        info = self.client.get_collection("test_games")
        self.assertEqual(info.config.params.vectors.size, 3)
        self.assertEqual(info.config.params.vectors.distance, models.Distance.COSINE)
        self.assertEqual(self.client.count("test_games", exact=True).count, 2)

    def test_repeat_upload_reuses_ids_and_updates_payload_without_changing_files(self):
        self.run_ingest(create_collection=True)
        self.write_games(price="25.0")
        before = (self.games.read_bytes(), self.descriptors.read_bytes())
        self.assertEqual(self.run_ingest(create_collection=True), 2)
        self.assertEqual(self.client.count("test_games", exact=True).count, 2)
        point = self.client.retrieve("test_games", ids=[10], with_vectors=True)[0]
        self.assertEqual(point.payload["Price"], "25.0")
        self.assertEqual(point.payload["Description"], "A game description.")
        self.assertEqual(point.payload["descriptors"], ["Adventure", "Exploration", "Single-player"])
        self.assertEqual(len(point.vector), 3)
        self.assertEqual(before, (self.games.read_bytes(), self.descriptors.read_bytes()))

    def test_existing_unrelated_points_are_not_deleted(self):
        self.create_collection()
        self.client.upsert("test_games", [models.PointStruct(id=99, vector=[1, 0, 0], payload={"Name": "Existing game"})])
        self.run_ingest(create_collection=True)
        self.assertEqual(self.client.count("test_games", exact=True).count, 3)
        self.assertEqual(self.client.retrieve("test_games", ids=[99])[0].payload["Name"], "Existing game")

    def test_incompatible_dimensions_stop_without_modifying_collection(self):
        self.create_collection(size=2)
        with self.assertRaisesRegex(QdrantConnectorError, "vector size"):
            self.run_ingest(create_collection=True)
        self.assertEqual(self.client.get_collection("test_games").config.params.vectors.size, 2)
        self.assertEqual(self.client.count("test_games", exact=True).count, 0)

    def test_wrong_distance_stops_without_modifying_collection(self):
        self.create_collection(distance=models.Distance.DOT)
        with self.assertRaisesRegex(QdrantConnectorError, "expected Cosine"):
            self.run_ingest(create_collection=True)
        self.assertEqual(self.client.get_collection("test_games").config.params.vectors.distance, models.Distance.DOT)
        self.assertEqual(self.client.count("test_games", exact=True).count, 0)

    def test_even_a_single_named_vector_is_rejected(self):
        self.create_collection(named=True)
        with self.assertRaisesRegex(QdrantConnectorError, "named vector"):
            self.run_ingest(create_collection=True)
        self.assertEqual(self.client.count("test_games", exact=True).count, 0)

    def test_mismatched_ids_fail_before_database_access(self):
        self.write_descriptors(rows=[[10, 1, 0, 0]])
        with self.assertRaisesRegex(ValueError, "no descriptor row"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()
        self.get_client.assert_not_called()

    def test_duplicate_game_ids_fail_before_database_access(self):
        self.write_games(ids=[10, 10])
        with self.assertRaisesRegex(ValueError, "Duplicate AppID"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_duplicate_descriptor_ids_fail_before_database_access(self):
        self.write_descriptors(rows=[[10, 1, 0, 0], [10, 0, 1, 0]])
        with self.assertRaisesRegex(ValueError, "Duplicate AppID"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_missing_payload_columns_fail_before_database_access(self):
        self.write_games(headers=["AppID", "Name"])
        with self.assertRaisesRegex(ValueError, "Missing required game columns"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_duplicate_headers_fail_before_database_access(self):
        self.write_descriptors(headers=["AppID", "Adventure", "Adventure", "Single-player"])
        with self.assertRaisesRegex(ValueError, "duplicate column names"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_malformed_descriptor_rows_fail_before_database_access(self):
        self.write_descriptors(rows=[[10, 1, 0], [20, 1, 0, 1]])
        with self.assertRaisesRegex(ValueError, "column count"):
            self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_nonfinite_and_nonnumeric_vectors_fail_before_database_access(self):
        for value in ["nan", "inf", "not-a-number"]:
            with self.subTest(value=value):
                self.write_descriptors(rows=[[10, value, 1, 0], [20, 1, 0, 1]])
                with self.assertRaises(ValueError):
                    self.run_ingest(create_collection=True)
        self.check.assert_not_called()

    def test_cli_validation_and_mutually_exclusive_actions(self):
        args = ["--games-path", str(self.games), "--descriptors-path", str(self.descriptors)]
        self.assertEqual(ingest.main([*args, "--validate-only"]), 0)
        with patch("sys.stderr", new_callable=io.StringIO), self.assertRaises(SystemExit) as caught:
            ingest.main([*args, "--validate-only", "--create-collection"])
        self.assertEqual(caught.exception.code, 2)
        self.check.assert_not_called()

    def test_invalid_batch_size_is_rejected_before_database_access(self):
        with self.assertRaisesRegex(ValueError, "batch_size must be positive"):
            self.run_ingest(batch_size=0, create_collection=True)
        self.check.assert_not_called()


if __name__ == "__main__":
    unittest.main()
