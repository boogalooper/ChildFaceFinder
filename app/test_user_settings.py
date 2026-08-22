from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from user_settings import load_user_settings, save_user_settings


class UserSettingsTests(unittest.TestCase):
    def test_roundtrip_preserves_unicode_and_types(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            expected = {"search_mode": "Улучшенный", "recursive": True, "workers": 4}
            save_user_settings(expected, path)
            self.assertEqual(load_user_settings(path), expected)

    def test_invalid_file_falls_back_to_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            path.write_text("not json", encoding="utf-8")
            self.assertEqual(load_user_settings(path), {})



if __name__ == "__main__":
    unittest.main()
