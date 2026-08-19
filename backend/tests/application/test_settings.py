from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from memorywaveform_extractor.application.settings import Settings


class SettingsTests(unittest.TestCase):
    def test_environment_origins_are_trimmed_for_the_local_web_ui(self) -> None:
        """The browser client needs explicit local origins instead of a permissive wildcard."""
        with patch.dict(
            os.environ,
            {"CORS_ORIGINS": " http://localhost:5173, http://127.0.0.1:4173 ,,"},
        ):
            settings = Settings.from_environment()

        self.assertEqual(
            settings.cors_origins,
            ("http://localhost:5173", "http://127.0.0.1:4173"),
        )


if __name__ == "__main__":
    unittest.main()
