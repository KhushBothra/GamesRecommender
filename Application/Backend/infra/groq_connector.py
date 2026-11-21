from __future__ import annotations

from Backend.infra.config import settings


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

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        extra_body = None
        if self.model.startswith("openai/gpt-oss-"):
            # The default medium reasoning effort can consume a small completion
            # budget before GPT-OSS emits user-facing content. These tasks need
            # concise, schema-bound output, so use low effort and omit the
            # separate reasoning field from the response.
            # extra_body works with the installed Groq SDK as well as newer
            # releases whose typed method signature includes these fields.
            extra_body = {"reasoning_effort": "low", "include_reasoning": False}

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        content = response.choices[0].message.content
        if not content:
            raise RuntimeError("Groq returned an empty response.")
        return content
