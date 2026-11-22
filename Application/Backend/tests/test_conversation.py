from __future__ import annotations

import copy
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi.testclient import TestClient

from Backend import main
from Backend.orchestrator import ApplicationOrchestrator
from Backend.tasks.conversation.schemas import DECISION_JSON_SCHEMA, ResponseSchemaError
from Backend.tasks.conversation.service import ConversationProtocolError, ConversationService
from Backend.tasks.conversation.stateful import ConversationStore


INSUFFICIENT = {"class": 4, "reason": "INSUFFICIENT_QUERY_CONTEXT"}
QUESTION = {"class": 1, "question": "Which game mechanics do you enjoy?"}


def allowed_classes(schema):
    return {
        branch["properties"]["class"]["enum"][0]
        for branch in schema["anyOf"]
    }


class ConversationStateTests(unittest.TestCase):
    def setUp(self):
        self.service = ConversationService()
        self.service.store = ConversationStore()

    def session(self, question_count):
        session = self.service.store.create()
        session.append("user", "I want horror games.")
        for number in range(question_count):
            session.append("assistant", f"Preference question {number + 1}?")
            session.append("user", "No preference.")
        session.clarification_questions_asked = question_count
        return session

    def test_early_refusal_is_unavailable_through_fifth_question(self):
        original_schema = copy.deepcopy(DECISION_JSON_SCHEMA)
        for count in range(5):
            with self.subTest(count=count):
                session = self.session(count)
                history = copy.deepcopy(session.messages)
                with patch("Backend.tasks.conversation.service.chat", return_value=json.dumps(QUESTION)) as chat:
                    turn = self.service._advance(session)
                self.assertEqual(turn.status, "needs_input")
                self.assertEqual(session.clarification_questions_asked, count + 1)
                self.assertEqual(allowed_classes(chat.call_args.kwargs["response_schema"]), {1, 2})
                messages = chat.call_args.args[0]
                system = "\n".join(m["content"] for m in messages if m["role"] == "system")
                self.assertIn(f"Clarification questions already asked: {count}", system)
                self.assertIn(f"Clarification questions remaining: {5 - count}", system)
                self.assertEqual([m for m in messages if m["role"] != "system"], history)
        self.assertEqual(DECISION_JSON_SCHEMA, original_schema)

    def test_after_fifth_answer_refusal_is_valid_and_sixth_question_is_unavailable(self):
        session = self.session(5)
        with patch("Backend.tasks.conversation.service.chat", return_value=json.dumps(INSUFFICIENT)) as chat:
            turn = self.service._advance(session)
        self.assertEqual(turn.status, "refused")
        self.assertEqual(allowed_classes(chat.call_args.kwargs["response_schema"]), {2, 4})
        self.assertEqual(session.clarification_questions_asked, 5)
        self.assertIsNone(self.service.store.get(session.conversation_id))

    def test_early_refusal_repair_has_exact_count_and_same_restricted_schema(self):
        session = self.session(3)
        with patch("Backend.tasks.conversation.service.chat", side_effect=[json.dumps(INSUFFICIENT), json.dumps(QUESTION)]) as chat:
            turn = self.service._advance(session)
        self.assertEqual(turn.status, "needs_input")
        self.assertEqual(chat.call_count, 2)
        first, repair = chat.call_args_list
        self.assertEqual(allowed_classes(repair.kwargs["response_schema"]), {1, 2})
        self.assertEqual(first.kwargs["response_schema"], repair.kwargs["response_schema"])
        self.assertEqual(repair.args[0][:-1], first.args[0])
        self.assertIn("Clarification questions already asked: 3", repair.args[0][-1]["content"])
        self.assertIn("Clarification questions remaining: 2", repair.args[0][-1]["content"])
        self.assertEqual(session.clarification_questions_asked, 4)
        self.assertEqual(len(session.messages), 8)

    def test_sixth_question_repair_keeps_terminal_schema(self):
        session = self.session(5)
        with patch("Backend.tasks.conversation.service.chat", side_effect=[json.dumps(QUESTION), json.dumps(INSUFFICIENT)]) as chat:
            turn = self.service._advance(session)
        self.assertEqual(turn.status, "refused")
        for request in chat.call_args_list:
            self.assertEqual(allowed_classes(request.kwargs["response_schema"]), {2, 4})
        self.assertEqual(session.clarification_questions_asked, 5)

    def test_runtime_validation_still_rejects_illegal_outcomes(self):
        with self.assertRaises(ResponseSchemaError):
            self.service._validate_payload(INSUFFICIENT, self.session(4))
        with self.assertRaises(ResponseSchemaError):
            self.service._validate_payload(QUESTION, self.session(5))
        with patch("Backend.tasks.conversation.service.chat", return_value=json.dumps(INSUFFICIENT)) as chat:
            with self.assertRaises(ConversationProtocolError):
                self.service._advance(self.session(3))
        self.assertEqual(chat.call_count, 2)

    def test_api_continues_until_five_answers_when_provider_obeys_schema(self):
        def provider(messages, *, response_schema, **kwargs):
            # Reproduce a model that chooses refusal whenever its schema permits
            # it. The old unrestricted schema fails even at the opening turn.
            if 4 in allowed_classes(response_schema):
                return json.dumps(INSUFFICIENT)
            number = 1 + sum(m["role"] == "assistant" for m in messages)
            return json.dumps({"class": 1, "question": f"Preference question {number}?"})

        with patch.object(main, "orchestrator", ApplicationOrchestrator(self.service)), patch(
            "Backend.tasks.conversation.service.chat", side_effect=provider
        ) as chat, TestClient(main.app) as api:
            response = api.post("/api/v1/conversations", json={"user_text": "I want horror games."})
            self.assertEqual(response.status_code, 200)
            conversation_id = response.json()["conversation_id"]
            for count in range(1, 6):
                self.assertEqual(response.json()["status"], "needs_input")
                self.assertEqual(response.json()["message"], f"Preference question {count}?")
                response = api.post(
                    f"/api/v1/conversations/{conversation_id}/turns",
                    json={"user_text": "No preference."},
                )
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["conversation_id"], conversation_id)
            self.assertEqual(response.json()["status"], "complete")
            self.assertEqual(response.json()["recommendations"], [])
            self.assertIn("couldn't gather enough", response.json()["message"])
        self.assertEqual(chat.call_count, 6)
        self.assertIsNone(self.service.store.get(conversation_id))


if __name__ == "__main__":
    unittest.main()
