"""Adapter for a locally running Ollama vision model."""

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


class OllamaVisionProvider:
    """Calls Ollama's local chat API and asks it for schema-shaped JSON."""

    def __init__(self, base_url: str, model: str, client: JsonHttpClient | None = None) -> None:
        if not base_url.strip():
            raise ValueError("Ollama base URL must not be empty.")
        if not model.strip():
            raise ValueError("Ollama model name must not be empty.")
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._client = client or UrllibJsonClient()

    @classmethod
    def from_environment(cls) -> OllamaVisionProvider:
        return cls(
            base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
            model=os.environ.get("OLLAMA_VISION_MODEL", os.environ.get("VISION_MODEL", "")),
        )

    def extract(
        self,
        image: bytes,
        schema: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "user",
                    "content": provider_prompt(schema, context),
                    "images": [base64.b64encode(image).decode("ascii")],
                }
            ],
        }
        response = self._client.post_json(
            f"{self._base_url}/api/chat", payload, {"Content-Type": "application/json"}
        )
        message = response.get("message")
        if not isinstance(message, dict) or "content" not in message:
            raise VisionProviderError("Ollama response did not contain message content.")
        return parse_json_object(message["content"])
