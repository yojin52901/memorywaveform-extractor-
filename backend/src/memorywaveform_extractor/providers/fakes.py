"""Deterministic in-memory adapters for tests and offline development."""

from __future__ import annotations

from copy import deepcopy


class FakeVisionProvider:
    """Returns a fixed payload while retaining call evidence for assertions."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = deepcopy(payload)
        self.call_count = 0
        self.last_image: bytes | None = None
        self.last_schema: dict[str, object] | None = None
        self.last_context: dict[str, object] | None = None

    def extract(
        self,
        image: bytes,
        schema: dict[str, object],
        context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        self.call_count += 1
        self.last_image = image
        self.last_schema = deepcopy(schema)
        self.last_context = deepcopy(context)
        return deepcopy(self._payload)
