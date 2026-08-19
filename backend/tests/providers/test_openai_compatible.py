from __future__ import annotations

import base64
import json
import unittest

from memorywaveform_extractor.providers.ollama import OllamaVisionProvider
from memorywaveform_extractor.providers.openai_compatible import OpenAICompatibleVisionProvider


class RecordingJsonClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.requests: list[tuple[str, dict[str, object], dict[str, str]]] = []

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        self.requests.append((url, payload, headers))
        return self.response


class OpenAICompatibleProviderTests(unittest.TestCase):
    def test_provider_sends_image_as_data_url_to_openai_compatible_endpoint(self) -> None:
        """A raw image blob would not be understood by an OpenAI-compatible chat endpoint."""
        client = RecordingJsonClient(
            {"choices": [{"message": {"content": json.dumps({"signals": []})}}]}
        )
        provider = OpenAICompatibleVisionProvider(
            base_url="http://localhost:8080",
            model="local-vlm",
            client=client,
        )

        result = provider.extract(b"png-bytes", {"type": "object"})

        self.assertEqual(result, {"signals": []})
        url, payload, headers = client.requests[0]
        self.assertEqual(url, "http://localhost:8080/v1/chat/completions")
        self.assertEqual(headers["Content-Type"], "application/json")
        self.assertEqual(payload["model"], "local-vlm")
        messages = payload["messages"]
        assert isinstance(messages, list)
        content = messages[0]["content"]
        assert isinstance(content, list)
        image_url = content[1]["image_url"]["url"]
        self.assertEqual(
            image_url,
            "data:image/png;base64," + base64.b64encode(b"png-bytes").decode("ascii"),
        )

    def test_ollama_provider_sends_base64_image_and_schema_format(self) -> None:
        """Ollama expects image data in its message image list rather than an OpenAI data URL."""
        client = RecordingJsonClient(
            {"message": {"content": json.dumps({"timing_parameters": []})}}
        )
        provider = OllamaVisionProvider(
            base_url="http://localhost:11434/",
            model="qwen-local",
            client=client,
        )
        schema = {"type": "object", "properties": {"timing_parameters": {"type": "array"}}}

        result = provider.extract(b"png-bytes", schema)

        self.assertEqual(result, {"timing_parameters": []})
        url, payload, _ = client.requests[0]
        self.assertEqual(url, "http://localhost:11434/api/chat")
        self.assertEqual(payload["model"], "qwen-local")
        self.assertEqual(payload["format"], schema)
        messages = payload["messages"]
        assert isinstance(messages, list)
        self.assertEqual(
            messages[0]["images"],
            [base64.b64encode(b"png-bytes").decode("ascii")],
        )


if __name__ == "__main__":
    unittest.main()
