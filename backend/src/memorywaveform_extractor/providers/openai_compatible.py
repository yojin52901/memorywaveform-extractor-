"""Adapter for local MLX and vLLM servers exposing OpenAI-compatible chat APIs."""

from __future__ import annotations

import base64
import os

from memorywaveform_extractor.providers.http import (
    JsonHttpClient,
    UrllibJsonClient,
    VisionProviderError,
    parse_json_object,
    provider_prompt,
)


class OpenAICompatibleVisionProvider:
    """Calls an OpenAI-compatible local vision endpoint with a PNG data URL."""

    def __init__(self, base_url: str, model: str, client: JsonHttpClient | None = None) -> None:
        if not base_url.strip():
            raise ValueError("OpenAI-compatible base URL must not be empty.")
        if not model.strip():
            raise ValueError("OpenAI-compatible model name must not be empty.")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or UrllibJsonClient()

    @classmethod
    def from_environment(cls) -> OpenAICompatibleVisionProvider:
        return cls(
            base_url=os.environ.get("OPENAI_COMPATIBLE_BASE_URL", "http://localhost:8000"),
            model=os.environ.get("OPENAI_COMPATIBLE_MODEL", os.environ.get("VISION_MODEL", "")),
        )

    def extract(
        self,
        image: bytes,
        schema: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        image_data_url = "data:image/png;base64," + base64.b64encode(image).decode("ascii")
        payload: dict[str, object] = {
            "model": self._model,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": provider_prompt(schema, context)},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
        }
        response = self._client.post_json(
            self._chat_completion_url(), payload, {"Content-Type": "application/json"}
        )
        return parse_json_object(_openai_message_content(response))

    def _chat_completion_url(self) -> str:
        if self._base_url.endswith("/v1"):
            return f"{self._base_url}/chat/completions"
        return f"{self._base_url}/v1/chat/completions"


def _openai_message_content(response: dict[str, object]) -> object:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise VisionProviderError("OpenAI-compatible response did not contain choices.")
    first_choice = choices[0]
    if not isinstance(first_choice, dict):
        raise VisionProviderError("OpenAI-compatible response choice was not an object.")
    message = first_choice.get("message")
    if not isinstance(message, dict) or "content" not in message:
        raise VisionProviderError("OpenAI-compatible response did not contain message content.")
    return message["content"]
