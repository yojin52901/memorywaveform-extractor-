"""Small standard-library JSON HTTP adapter shared by local providers."""

from __future__ import annotations

import json
from typing import Protocol
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class VisionProviderError(RuntimeError):
    """Raised when a local VLM endpoint returns an unusable response."""


class JsonHttpClient(Protocol):
    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        """POST one JSON object and decode one JSON-object response."""


class UrllibJsonClient:
    """A dependency-free HTTP client for local-only model endpoints."""

    def __init__(self, timeout_seconds: float = 120.0) -> None:
        self._timeout_seconds = timeout_seconds

    def post_json(
        self,
        url: str,
        payload: dict[str, object],
        headers: dict[str, str],
    ) -> dict[str, object]:
        request = Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urlopen(request, timeout=self._timeout_seconds) as response:  # noqa: S310
                response_body = response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError, OSError) as error:
            raise VisionProviderError(f"Local vision endpoint request failed: {error}") from error

        try:
            decoded = json.loads(response_body)
        except json.JSONDecodeError as error:
            raise VisionProviderError("Local vision endpoint returned invalid JSON.") from error
        if not isinstance(decoded, dict):
            raise VisionProviderError("Local vision endpoint returned a non-object JSON response.")
        return decoded


def provider_prompt(schema: dict[str, object], context: dict[str, object] | None = None) -> str:
    """Keep model instructions explicit without embedding any external domain glossary."""

    prompt = (
        "Read the supplied memory timing-diagram image. Extract only information visible in "
        "the image and return one JSON object matching this schema. Do not add Markdown, "
        "explanations, or facts that are not grounded in the image. Schema: "
        + json.dumps(schema, separators=(",", ":"))
    )
    if context is not None:
        prompt += (
            " Use this deterministic OCR and geometry candidate graph as grounding evidence. "
            "Only reuse its signal IDs, event IDs, timing IDs, and arrow endpoints: "
            + json.dumps(context, separators=(",", ":"))
        )
    return prompt


def parse_json_object(content: object) -> dict[str, object]:
    """Normalize a provider message content value to the contract payload object."""

    if isinstance(content, dict):
        return content
    if not isinstance(content, str):
        raise VisionProviderError("Local vision endpoint did not return textual JSON content.")

    normalized_content = content.strip()
    if normalized_content.startswith("```"):
        normalized_content = _strip_markdown_fence(normalized_content)
    try:
        parsed = json.loads(normalized_content)
    except json.JSONDecodeError as error:
        raise VisionProviderError("Local vision model returned invalid JSON content.") from error
    if not isinstance(parsed, dict):
        raise VisionProviderError("Local vision model returned JSON that was not an object.")
    return parsed


def _strip_markdown_fence(content: str) -> str:
    lines = content.splitlines()
    if len(lines) < 2 or not lines[-1].strip().startswith("```"):
        return content
    return "\n".join(lines[1:-1]).strip()
