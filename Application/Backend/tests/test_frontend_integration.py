from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from Backend.infra.config import BACKEND_DIR, get_settings
from Backend.infra.steam_artwork import (
    FAILURE_TTL_SECONDS, SUCCESS_TTL_SECONDS, SteamArtwork, _fetch_image,
)
from Backend.main import app
from Backend.orchestrator import ApplicationOrchestrator, _frontend_game
from Backend.tasks.conversation.service import ConversationNotFoundError, ConversationTurn


def game() -> dict:
    return {"appid": 620, "name": "Portal 2", "developers": ["Valve"], "publishers": ["Valve"], "price": "0.0", "description": "A generated explanation."}


class ArtworkTests(unittest.TestCase):
    def test_success_cache_expires_after_a_day(self):
        now = [0.0]
        fetcher = Mock(return_value="https://shared.fastly.steamstatic.com/header.jpg")
        client = SteamArtwork(fetcher, lambda: now[0])
        self.assertEqual(client.lookup(620), client.lookup(620))
        fetcher.assert_called_once_with(620)
        now[0] = SUCCESS_TTL_SECONDS + 1
        client.lookup(620)
        self.assertEqual(fetcher.call_count, 2)

    def test_failed_lookups_are_optional_and_cached_for_ten_minutes(self):
        now = [0.0]
        fetcher = Mock(side_effect=TimeoutError("Steam timeout"))
        client = SteamArtwork(fetcher, lambda: now[0])
        self.assertIsNone(client.lookup(620))
        self.assertIsNone(client.lookup(620))
        fetcher.assert_called_once()
        now[0] = FAILURE_TTL_SECONDS + 1
        self.assertIsNone(client.lookup(620))
        self.assertEqual(fetcher.call_count, 2)

    def test_enrichment_preserves_price_explanation_and_order(self):
        original = [game(), {**game(), "appid": 413150, "price": None}]
        client = SteamArtwork(lambda appid: f"https://shared.fastly.steamstatic.com/{appid}.jpg")
        enriched = client.enrich(original)
        self.assertEqual([g["appid"] for g in enriched], [620, 413150])
        self.assertEqual(enriched[0]["price"], "0.0")
        self.assertEqual(enriched[0]["description"], original[0]["description"])
        self.assertIsNone(enriched[1]["price"])
        self.assertNotIn("image_url", original[0])

    def test_only_the_header_image_is_extracted_from_steam(self):
        image = "https://shared.fastly.steamstatic.com/store_item_assets/steam/apps/620/header.jpg"
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps({"620": {"success": True, "data": {"header_image": image, "price_overview": {"final": 99}, "short_description": "Not our explanation"}}}).encode()
        with patch("Backend.infra.steam_artwork.urlopen", return_value=response) as fetch:
            self.assertEqual(_fetch_image(620), image)
            self.assertEqual(fetch.call_args.kwargs["timeout"], 3)
        response.read.return_value = json.dumps({"620": {"success": True, "data": {"header_image": "https://untrusted.example/header.jpg"}}}).encode()
        with patch("Backend.infra.steam_artwork.urlopen", return_value=response):
            self.assertIsNone(_fetch_image(620))

    def test_invalid_appids_do_not_make_network_calls(self):
        fetcher = Mock()
        client = SteamArtwork(fetcher)
        for appid in [0, -1, True, "620"]:
            self.assertIsNone(client.lookup(appid))
        fetcher.assert_not_called()


class ApiContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_start_and_turn_use_user_text_and_preserve_conversation_id(self):
        first = {"conversation_id": "session-1", "status": "needs_input", "message": "Single-player or multiplayer?", "recommendations": []}
        final = {**first, "status": "complete", "message": "Games for you.", "recommendations": [{**game(), "image_url": None}]}
        with patch("Backend.main.orchestrator.start_conversation", return_value=first) as start, patch("Backend.main.orchestrator.answer_cross_question", return_value=final) as answer:
            self.assertEqual(self.client.post("/api/v1/conversations", json={"user_text": "An exploration game."}).json(), first)
            result = self.client.post("/api/v1/conversations/session-1/turns", json={"user_text": "Single-player."})
            self.assertEqual(result.json(), final)
            start.assert_called_once_with("An exploration game.")
            answer.assert_called_once_with("session-1", "Single-player.")

    def test_expired_and_invalid_requests_return_expected_statuses(self):
        with patch("Backend.main.orchestrator.answer_cross_question", side_effect=ConversationNotFoundError("Expired")):
            self.assertEqual(self.client.post("/api/v1/conversations/expired/turns", json={"user_text": "An answer."}).status_code, 404)
        self.assertEqual(self.client.post("/api/v1/conversations", json={"user_text": ""}).status_code, 422)
        self.assertEqual(self.client.post("/api/v1/conversations", json={"user_text": "a" * 4001}).status_code, 422)

    def test_health_is_liveness_without_calling_a_model(self):
        result = self.client.get("/api/v1/health")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.json()["status"], "ok")

    def test_missing_image_remains_backward_compatible(self):
        final = {"conversation_id": "direct", "status": "complete", "message": "Ready.", "recommendations": [game()]}
        with patch("Backend.main.orchestrator.start_conversation", return_value=final):
            result = self.client.post("/api/v1/conversations", json={"user_text": "An exploration game."})
        self.assertEqual(result.status_code, 200)
        self.assertIsNone(result.json()["recommendations"][0]["image_url"])

    def test_descriptor_default_resolves_to_repository_data_prep(self):
        with patch.dict("os.environ", {}, clear=True):
            settings = get_settings()
        self.assertEqual(settings.games_descriptors_path, BACKEND_DIR.parents[1] / "Data Prep" / "games_descriptors.csv")

    def test_payload_mapping_does_not_treat_numeric_zero_as_free(self):
        result = _frontend_game({"payload": {"AppID": 620, "Name": "Portal 2", "Developers": '["Valve", "Valve Software"]', "Publishers": "Valve", "Price": 0.0}, "explanation": "An explanation."})
        self.assertEqual(result["price"], "0.0")
        self.assertEqual(result["developers"], ["Valve", "Valve Software"])

    def test_artwork_failure_does_not_fail_the_orchestrator(self):
        conversations = Mock()
        conversations.store.get.return_value = Mock(messages=[{"role": "user", "content": "Explore"}])
        from threading import RLock
        conversations.store.get.return_value.lock = RLock()
        presenter = Mock()
        presenter.generate.return_value = [{"payload": {"AppID": 620, "Name": "Portal 2", "Price": None}, "explanation": "An explanation."}]
        orchestrator = ApplicationOrchestrator(conversations, presenter)
        turn = ConversationTurn("id", "query_ready", "Ready", {"tags": ["Exploration"], "genres": [], "categories": []})
        with patch("Backend.orchestrator.recommend_games", return_value=[]), patch("Backend.orchestrator.steam_artwork.enrich", side_effect=RuntimeError("Artwork failure")):
            result = orchestrator._handle_turn(turn)
        self.assertEqual(result["status"], "complete")
        self.assertIsNone(result["recommendations"][0]["image_url"])


if __name__ == "__main__":
    unittest.main()
