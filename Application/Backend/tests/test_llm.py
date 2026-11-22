from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import httpx
from groq import APIStatusError, BadRequestError, Groq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Backend.infra import llm
from Backend.infra.groq_connector import GroqConnector
from Backend.tasks.conversation.schemas import DECISION_JSON_SCHEMA, parse_decision
from Backend.tasks.output.schemas import OUTPUT_JSON_SCHEMA, parse_output


QUESTION = {"class": 1, "question": "Do you prefer a modern or historical setting?"}
TOOL_ERROR = {
    "error": {
        "message": "Tool choice is none, but model called a tool",
        "type": "invalid_request_error",
        "code": "tool_use_failed",
        "failed_generation": json.dumps({"name": "assistant", "arguments": QUESTION}),
    }
}
JSON_ERROR = {
    "error": {
        "message": "Failed to validate JSON. Please adjust your prompt.",
        "type": "invalid_request_error",
        "code": "json_validate_failed",
        "failed_generation": "",
    }
}


def completion(content: str | None, finish_reason: str = "stop") -> dict:
    return {
        "id": "test-completion",
        "object": "chat.completion",
        "created": 1,
        "model": "openai/gpt-oss-20b",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": {"role": "assistant", "content": content}}],
    }


def structured_completion(payload: dict) -> dict:
    return completion(json.dumps({"response": payload}))


class GroqConnectorTests(unittest.TestCase):
    def setUp(self):
        self.requests = []
        self.responses = []

        def handle(request):
            self.requests.append(json.loads(request.content))
            status, body = self.responses.pop(0)
            return httpx.Response(status, json=body)

        self.client = Groq(
            api_key="test-key",
            max_retries=0,
            http_client=httpx.Client(transport=httpx.MockTransport(handle)),
        )
        self.addCleanup(self.client.close)
        with patch("groq.Groq", return_value=self.client):
            self.connector = GroqConnector(api_key="test-key")
        self.connector.model = "openai/gpt-oss-20b"
        self.messages = [
            {"role": "system", "content": "Return the decision as JSON."},
            {"role": "user", "content": "I enjoy exploration games."},
        ]

    def chat(self):
        return self.connector.chat(self.messages, temperature=0.2, max_tokens=700)

    def test_json_mode_preserves_history_and_reasoning_budget(self):
        original = copy.deepcopy(self.messages)
        self.responses = [(200, completion(json.dumps(QUESTION)))]
        self.assertEqual(json.loads(self.chat()), QUESTION)
        request = self.requests[0]
        self.assertEqual(request["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", request)
        self.assertNotIn("tool_choice", request)
        self.assertEqual(request["reasoning_effort"], "low")
        self.assertFalse(request["include_reasoning"])
        self.assertEqual(request["max_completion_tokens"], 2048)
        self.assertNotIn("max_tokens", request)
        self.assertEqual(request["messages"][1:], original[1:])
        self.assertTrue(request["messages"][0]["content"].startswith(original[0]["content"]))
        self.assertEqual(self.messages, original)

    def test_tool_error_recovers_with_same_key(self):
        self.responses = [(400, TOOL_ERROR), (200, completion(json.dumps(QUESTION)))]
        config = SimpleNamespace(groq_api_keys=("test-key",))
        with patch.object(llm, "settings", config), patch.object(
            llm, "GroqConnector", return_value=self.connector
        ):
            answer = llm.chat(self.messages, max_tokens=700)
        self.assertEqual(json.loads(answer), QUESTION)
        self.assertEqual(len(self.requests), 2)
        retry = self.requests[1]
        self.assertEqual(retry["temperature"], 0)
        self.assertEqual(retry["max_completion_tokens"], 4096)
        self.assertEqual(retry["response_format"], {"type": "json_object"})
        self.assertIn("previous generation was rejected", retry["messages"][0]["content"])
        self.assertEqual(retry["messages"][1:], self.messages[1:])

    def test_strict_decision_schema_handles_every_response_class(self):
        original_messages = copy.deepcopy(self.messages)
        original_schema = copy.deepcopy(DECISION_JSON_SCHEMA)
        payloads = [
            QUESTION,
            {"class": 2, "query": {"tags": ["Horror"], "genres": [], "categories": []}},
            {"class": 4, "reason": "INSUFFICIENT_QUERY_CONTEXT"},
        ]
        for model in ("openai/gpt-oss-20b", "openai/gpt-oss-120b"):
            for payload in payloads:
                with self.subTest(model=model, payload=payload):
                    self.connector.model = model
                    self.responses = [(200, structured_completion(payload))]
                    result = self.connector.chat(
                        self.messages, temperature=0.2, max_tokens=700,
                        response_schema=DECISION_JSON_SCHEMA,
                    )
                    self.assertEqual(json.loads(result), payload)
                    self.assertEqual(parse_decision(json.loads(result)).response_class, payload["class"])
                    request = self.requests[-1]
                    self.assertEqual(request["response_format"]["type"], "json_schema")
                    specification = request["response_format"]["json_schema"]
                    self.assertTrue(specification["strict"])
                    self.assertEqual(specification["schema"]["properties"]["response"], original_schema)
                    self.assertNotIn("anyOf", specification["schema"])
                    self.assertEqual(request["messages"][1:], self.messages[1:])
                    self.assertIn("Transport format:", request["messages"][0]["content"])
                    self.assertNotIn("tool_choice", request)
        self.assertEqual(self.messages, original_messages)
        self.assertEqual(DECISION_JSON_SCHEMA, original_schema)

    def test_strict_output_schema_preserves_game_order_ids_and_null_explanation(self):
        payload = {"games": [
            {"appid": 620, "explanation": "Puzzle exploration."},
            {"appid": 10, "explanation": None},
        ]}
        self.responses = [(200, structured_completion(payload))]
        result = self.connector.chat(
            self.messages, temperature=0, max_tokens=700,
            response_schema=OUTPUT_JSON_SCHEMA,
        )
        self.assertEqual(json.loads(result), payload)
        self.assertEqual([g.appid for g in parse_output(json.loads(result), [620, 10])], [620, 10])

    def test_structured_retry_keeps_strict_schema(self):
        self.responses = [(400, JSON_ERROR), (200, structured_completion(QUESTION))]
        result = self.connector.chat(
            self.messages, temperature=0.2, max_tokens=700,
            response_schema=DECISION_JSON_SCHEMA,
        )
        self.assertEqual(json.loads(result), QUESTION)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["response_format"], self.requests[1]["response_format"])
        self.assertEqual(self.requests[1]["max_completion_tokens"], 4096)

    def test_conversation_semantic_repair_keeps_schema_and_history(self):
        from Backend.tasks.conversation.service import ConversationService
        from Backend.tasks.conversation.stateful import ConversationStore

        invalid_query = {"class": 2, "query": {
            "tags": ["Not a real descriptor"], "genres": [], "categories": [],
        }}
        self.responses = [
            (200, structured_completion(invalid_query)),
            (200, structured_completion(QUESTION)),
        ]
        conversations = ConversationService()
        conversations.store = ConversationStore()
        with patch.object(llm, "settings", SimpleNamespace(groq_api_keys=("test-key",))), patch.object(
            llm, "GroqConnector", return_value=self.connector
        ):
            turn = conversations.start("I want horror games.")
        self.assertEqual(turn.status, "needs_input")
        self.assertEqual(turn.message, QUESTION["question"])
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["response_format"], self.requests[1]["response_format"])
        session = conversations.store.get(turn.conversation_id)
        self.assertEqual(session.clarification_questions_asked, 1)
        self.assertEqual(len(session.messages), 2)

    def test_output_semantic_repair_keeps_schema_and_original_game_data(self):
        from Backend.tasks.output.service import OutputService

        games = [{"rank": 1, "payload": {"AppID": 620, "Name": "Portal 2"}}]
        original = copy.deepcopy(games)
        self.responses = [
            (200, structured_completion({"games": [{"appid": 999, "explanation": "Wrong game."}]})),
            (200, structured_completion({"games": [{"appid": 620, "explanation": "Puzzle exploration."}]})),
        ]
        with patch.object(llm, "settings", SimpleNamespace(groq_api_keys=("test-key",))), patch.object(
            llm, "GroqConnector", return_value=self.connector
        ):
            result = OutputService().generate(
                [{"role": "user", "content": "I enjoy puzzle exploration."}],
                {"tags": ["Puzzle"], "genres": [], "categories": []},
                games,
            )
        self.assertEqual(result[0]["payload"], original[0]["payload"])
        self.assertEqual(result[0]["explanation"], "Puzzle exploration.")
        self.assertEqual(games, original)
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["response_format"], self.requests[1]["response_format"])
        specification = self.requests[1]["response_format"]["json_schema"]
        self.assertEqual(specification["schema"]["properties"]["response"], OUTPUT_JSON_SCHEMA)

    def test_invalid_structured_envelopes_are_not_returned_as_answers(self):
        for content in ('{}', '{"response": []}', '{"response": {}, "extra": 1}', 'not json'):
            with self.subTest(content=content):
                self.requests.clear()
                self.responses = [(200, completion(content))] * 2
                with self.assertRaisesRegex(RuntimeError, "invalid structured response"):
                    self.connector.chat(
                        self.messages, temperature=0, max_tokens=700,
                        response_schema=DECISION_JSON_SCHEMA,
                    )
                self.assertEqual(len(self.requests), 2)

    def test_unsupported_model_retains_json_mode_without_transport_wrapper(self):
        self.connector.model = "other-model"
        self.responses = [(200, completion(json.dumps(QUESTION)))]
        result = self.connector.chat(
            self.messages, temperature=0, max_tokens=700,
            response_schema=DECISION_JSON_SCHEMA,
        )
        self.assertEqual(json.loads(result), QUESTION)
        self.assertEqual(self.requests[0]["response_format"], {"type": "json_object"})
        self.assertNotIn("Transport format:", self.requests[0]["messages"][0]["content"])

    def test_repeated_tool_error_is_bounded_and_reports_groq_failure(self):
        self.responses = [(400, TOOL_ERROR), (400, TOOL_ERROR)]
        config = SimpleNamespace(groq_api_keys=("test-key",))
        with patch.object(llm, "settings", config), patch.object(
            llm, "GroqConnector", return_value=self.connector
        ):
            with self.assertRaisesRegex(llm.LLMError, "Groq chat failed") as error:
                llm.chat(self.messages)
        self.assertEqual(len(self.requests), 2)
        self.assertIsInstance(error.exception.__cause__, BadRequestError)
        self.assertIn("tool_use_failed", str(error.exception))

    def test_tool_error_with_flat_error_body_also_recovers(self):
        self.responses = [(400, TOOL_ERROR["error"]), (200, completion(json.dumps(QUESTION)))]
        self.assertEqual(json.loads(self.chat()), QUESTION)
        self.assertEqual(len(self.requests), 2)

    def test_empty_json_validation_failure_recovers(self):
        for error in (JSON_ERROR, JSON_ERROR["error"]):
            with self.subTest(error=error):
                self.requests.clear()
                self.responses = [(400, error), (200, completion(json.dumps(QUESTION)))]
                self.assertEqual(json.loads(self.chat()), QUESTION)
                self.assertEqual(len(self.requests), 2)
                self.assertEqual(self.requests[1]["max_completion_tokens"], 4096)

    def test_repeated_json_validation_failure_is_bounded(self):
        self.responses = [(400, JSON_ERROR), (400, JSON_ERROR)]
        with self.assertRaises(BadRequestError):
            self.chat()
        self.assertEqual(len(self.requests), 2)

    def test_larger_requested_budget_is_preserved_on_retry(self):
        self.responses = [(400, JSON_ERROR), (200, completion(json.dumps(QUESTION)))]
        self.connector.chat(self.messages, temperature=0, max_tokens=5000)
        self.assertEqual([r["max_completion_tokens"] for r in self.requests], [5000, 5000])

    def test_authentication_and_rate_limit_errors_are_not_generation_retries(self):
        for status in (401, 429):
            with self.subTest(status=status):
                self.requests.clear()
                self.responses = [(status, {"error": {"message": "Provider unavailable"}})]
                with self.assertRaises(APIStatusError):
                    self.chat()
                self.assertEqual(len(self.requests), 1)

    def test_other_bad_requests_are_not_retried(self):
        for body in ({"error": {"message": "Invalid model", "code": "model_not_found"}}, {"error": "Bad request"}):
            with self.subTest(body=body):
                self.requests.clear()
                self.responses = [(400, body)]
                with self.assertRaises(BadRequestError):
                    self.chat()
                self.assertEqual(len(self.requests), 1)

    def test_non_gpt_oss_models_omit_reasoning_options(self):
        self.connector.model = "other-model"
        self.responses = [(200, completion(json.dumps(QUESTION)))]
        self.chat()
        self.assertNotIn("reasoning_effort", self.requests[0])
        self.assertNotIn("include_reasoning", self.requests[0])
        self.assertEqual(self.requests[0]["max_completion_tokens"], 700)

    def test_empty_or_truncated_response_recovers(self):
        for content, finish in ((None, "stop"), ("   ", "stop"), ('{"class":', "length")):
            with self.subTest(content=content, finish=finish):
                self.requests.clear()
                self.responses = [
                    (200, completion(content, finish)),
                    (200, completion(json.dumps(QUESTION))),
                ]
                self.assertEqual(json.loads(self.chat()), QUESTION)
                self.assertEqual(len(self.requests), 2)

    def test_repeated_empty_response_still_fails(self):
        self.responses = [(200, completion(None)), (200, completion(None))]
        with self.assertRaisesRegex(RuntimeError, "empty response"):
            self.chat()
        self.assertEqual(len(self.requests), 2)

    def test_repeated_truncation_does_not_pass_partial_json_to_caller(self):
        self.responses = [(200, completion('{"class":', "length"))] * 2
        with self.assertRaisesRegex(RuntimeError, "truncated response"):
            self.chat()
        self.assertEqual(len(self.requests), 2)

    def test_later_api_turn_recovers_json_failure_without_losing_conversation(self):
        from fastapi.testclient import TestClient
        from Backend import main
        from Backend.orchestrator import ApplicationOrchestrator
        from Backend.tasks.conversation.service import ConversationService
        from Backend.tasks.conversation.stateful import ConversationStore

        questions = [
            "Do you prefer single-player or multiplayer?",
            "Do you prefer online or offline play?",
            "What setting or theme do you enjoy in horror games?",
            "Do you enjoy puzzles, exploration, or combat?",
        ]
        self.responses = [
            (200, structured_completion({"class": 1, "question": question}))
            for question in questions[:3]
        ] + [(400, JSON_ERROR), (200, structured_completion({"class": 1, "question": questions[3]}))]
        conversations = ConversationService()
        conversations.store = ConversationStore()
        config = SimpleNamespace(groq_api_keys=("test-key",))
        with patch.object(main, "orchestrator", ApplicationOrchestrator(conversations)), patch.object(
            llm, "settings", config
        ), patch.object(llm, "GroqConnector", return_value=self.connector), TestClient(main.app) as api:
            response = api.post("/api/v1/conversations", json={"user_text": "I want horror games."})
            self.assertEqual(response.status_code, 200)
            conversation_id = response.json()["conversation_id"]
            for answer in ("single-player", "offline", "psychological horror"):
                response = api.post(
                    f"/api/v1/conversations/{conversation_id}/turns",
                    json={"user_text": answer},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["conversation_id"], conversation_id)
            self.assertEqual(response.json()["status"], "needs_input")
            self.assertEqual(response.json()["message"], questions[3])
        self.assertEqual(len(self.requests), 5)
        self.assertTrue(all(r["response_format"]["type"] == "json_schema" for r in self.requests))
        self.assertEqual(self.requests[-1]["messages"][1:], self.requests[-2]["messages"][1:])
        self.assertEqual(conversations.store.get(conversation_id).clarification_questions_asked, 4)


class LLMChatTests(unittest.TestCase):
    def test_missing_keys_fail_before_constructing_a_connector(self):
        with patch.object(llm, "settings", SimpleNamespace(groq_api_keys=())), patch.object(
            llm, "GroqConnector"
        ) as connector:
            with self.assertRaisesRegex(llm.LLMError, "No GROQ_API_KEY"):
                llm.chat([{"role": "user", "content": "Horror games"}])
        connector.assert_not_called()

    def test_next_configured_key_is_tried_after_failure(self):
        config = SimpleNamespace(groq_api_keys=("first-key", "second-key"))
        first = Mock()
        first.chat.side_effect = RuntimeError("Rate limit exceeded")
        second = Mock()
        second.chat.return_value = json.dumps(QUESTION)
        messages = [{"role": "user", "content": "Horror games"}]
        with patch.object(llm, "settings", config), patch.object(
            llm, "GroqConnector", side_effect=[first, second]
        ) as connector:
            self.assertEqual(json.loads(llm.chat(messages)), QUESTION)
        self.assertEqual(connector.call_args_list, [call(api_key="first-key"), call(api_key="second-key")])
        second.chat.assert_called_once_with(messages, temperature=0.2, max_tokens=500)

    def test_all_failed_keys_report_errors_and_preserve_last_cause(self):
        config = SimpleNamespace(groq_api_keys=("first-key", "second-key"))
        first_error = RuntimeError("Rate limit exceeded")
        last_error = RuntimeError("JSON validation failed")
        with patch.object(llm, "settings", config), patch.object(
            llm, "GroqConnector", side_effect=[first_error, last_error]
        ) as connector:
            with self.assertRaises(llm.LLMError) as error:
                llm.chat([{"role": "user", "content": "Horror games"}])
        self.assertEqual(connector.call_count, 2)
        self.assertIn("key 1: Rate limit exceeded", str(error.exception))
        self.assertIn("key 2: JSON validation failed", str(error.exception))
        self.assertIs(error.exception.__cause__, last_error)


if __name__ == "__main__":
    unittest.main()
