from __future__ import annotations

import json
import logging
from typing import Any

from Backend.infra.config import settings

logger = logging.getLogger(__name__)

JSON_RESPONSE_INSTRUCTION = (
    "Return exactly one JSON object as the assistant message content, following "
    "the response schema in the task instructions. Tools and function calls are "
    "unavailable. Do not call a tool or wrap the response in a function name, "
    "arguments, Markdown, or explanatory text."
)


class GroqConnector:
    def __init__(self, api_key: str | None = None) -> None:
        api_key = api_key or (settings.groq_api_keys[0] if settings.groq_api_keys else None)
        if not api_key:
            raise RuntimeError("No GROQ_API_KEY or GROQ_API_KEY<n> value is set.")
        try:
            from groq import Groq
        except ImportError as exc:
            raise RuntimeError("Groq support is missing; install the 'groq' package.") from exc

        self.model = settings.groq_model
        self._client = Groq(api_key=api_key)

    def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        max_tokens: int,
        response_schema: dict[str, Any] | None = None,
    ) -> str:
        from groq import BadRequestError

        structured = response_schema is not None and self.model in {
            "openai/gpt-oss-20b", "openai/gpt-oss-120b",
        }
        response_format: dict[str, Any] = {"type": "json_object"}
        instruction = JSON_RESPONSE_INSTRUCTION
        if structured:
            # Keep a plain object at the schema root: the task's class 1/2/4
            # union belongs under a required field, not at the schema root.
            # This envelope is removed before existing task validation runs.
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": "task_response",
                    "strict": True,
                    "schema": {
                        "type": "object",
                        "properties": {"response": response_schema},
                        "required": ["response"],
                        "additionalProperties": False,
                    },
                },
            }
            instruction += (
                '\nTransport format: put the complete task JSON object inside '
                'a single top-level "response" field: {"response": <task JSON>}. '
                "The task's field and class rules apply inside response. "
                "This transport envelope is the only outer field."
            )

        def build_messages(instructions: str) -> list[dict[str, str]]:
            # Place transport instructions after the task's system instructions
            # so their respective inner/outer JSON formats do not conflict.
            system_end = 0
            while system_end < len(messages) and messages[system_end]["role"] == "system":
                system_end += 1
            return [
                {
                    "role": "system",
                    "content": "\n\n".join(
                        [message["content"] for message in messages[:system_end]]
                        + [instructions]
                    ),
                },
                *messages[system_end:],
            ]

        request_messages = build_messages(instruction)
        extra_body = None
        completion_budget = max_tokens
        if self.model.startswith("openai/gpt-oss-"):
            # The default medium reasoning effort can consume a small completion
            # budget before GPT-OSS emits user-facing content. These tasks need
            # concise, schema-bound output, so use low effort and omit the
            # separate reasoning field from the response.
            # extra_body works with the installed Groq SDK as well as newer
            # releases whose typed method signature includes these fields.
            extra_body = {"reasoning_effort": "low", "include_reasoning": False}
            # Hiding reasoning does not remove its token cost. A 700-token
            # budget can expire before any JSON is emitted, which Groq reports
            # as json_validate_failed with an empty failed_generation.
            completion_budget = max(max_tokens, 2048)

        for attempt in range(2):
            try:
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=request_messages,
                    temperature=temperature if attempt == 0 else 0,
                    max_completion_tokens=completion_budget,
                    response_format=response_format,
                    extra_body=extra_body,
                )
            except BadRequestError as exc:
                # JSON mode can reject an incomplete generation before our
                # task-level validator receives anything to repair. Never use
                # failed_generation as a successful answer.
                error = exc.body
                if isinstance(error, dict):
                    error = error.get("error", error)
                code = error.get("code") if isinstance(error, dict) else None
                if attempt or code not in {"tool_use_failed", "json_validate_failed"}:
                    raise
                retry_reason = code
            else:
                choice = response.choices[0]
                content = choice.message.content
                if choice.finish_reason == "length":
                    retry_reason = "truncated response"
                elif not content or not content.strip():
                    retry_reason = "empty response"
                elif structured:
                    try:
                        envelope = json.loads(content)
                        if (
                            not isinstance(envelope, dict)
                            or set(envelope) != {"response"}
                            or not isinstance(envelope["response"], dict)
                        ):
                            raise ValueError("Invalid structured response envelope")
                        return json.dumps(envelope["response"], ensure_ascii=False)
                    except ValueError:
                        retry_reason = "invalid structured response"
                else:
                    return content
                if attempt:
                    raise RuntimeError(f"Groq returned {retry_reason} after one retry.")

            completion_budget = max(completion_budget, 4096)
            logger.warning(
                "Groq returned %s; retrying once with %s and %d completion tokens.",
                retry_reason,
                response_format["type"],
                completion_budget,
            )
            request_messages = build_messages(
                instruction
                + " A previous generation was rejected because it did not produce "
                "a complete JSON answer. Keep reasoning brief and emit the complete "
                "JSON directly as your final answer."
            )

        raise RuntimeError("Groq did not produce a complete response.")
