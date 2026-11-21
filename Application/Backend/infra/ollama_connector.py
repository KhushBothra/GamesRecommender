from __future__ import annotations

import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from Backend.infra.config import settings


class OllamaConnector:
    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url
        self.chat_url = settings.ollama_chat_url
        self.chat_model = settings.ollama_model
        self.timeout = settings.ollama_timeout_seconds

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as result:
                response = json.loads(result.read().decode("utf-8"))
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Ollama request failed at {url}: {exc}") from exc
        if not isinstance(response, dict):
            raise RuntimeError("Ollama returned an invalid response.")
        return response

    def chat(self, messages: list[dict[str, str]], *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
                "num_gpu": settings.ollama_num_gpu,
            },
        }
        if self.chat_model.casefold().startswith("qwen3"):
            # Qwen3's default thinking mode can spend a bounded response budget
            # without producing user-facing content. This backend requires a
            # direct answer, especially for strict JSON responses.
            payload["think"] = False
        response = self._post(
            self.chat_url,
            payload,
        )
        content = response.get("message", {}).get("content")
        if not content:
            raise RuntimeError("Ollama returned an empty chat response.")
        content = str(content).strip()
        # Older Ollama builds may ignore the API's ``think`` flag and leave
        # Qwen3's reasoning trace in content, ending it with </think>. Never
        # pass that trace to the task validator or user-facing response.
        if "</think>" in content:
            content = content.rsplit("</think>", maxsplit=1)[-1].strip()
        elif "<think>" in content:
            raise RuntimeError("Ollama returned an unfinished reasoning block.")
        if not content:
            raise RuntimeError("Ollama returned no final answer after reasoning.")
        return content

