from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.runtime.dependencies import collect_missing_transcription_packages


class RuntimeDependencyTests(unittest.TestCase):
    @patch("whisper_sift.runtime.dependencies.shutil.which")
    @patch("whisper_sift.runtime.dependencies.importlib.util.find_spec")
    def test_system_ffmpeg_makes_imageio_optional(
        self,
        find_spec_mock,
        which_mock,
    ) -> None:
        which_mock.return_value = "/usr/bin/ffmpeg"

        def fake_find_spec(module_name: str) -> object | None:
            if module_name in {"torch", "whisper"}:
                return object()
            return None

        find_spec_mock.side_effect = fake_find_spec

        missing_packages = collect_missing_transcription_packages()

        self.assertEqual([], missing_packages)

    @patch("whisper_sift.runtime.dependencies.shutil.which")
    @patch("whisper_sift.runtime.dependencies.importlib.util.find_spec")
    def test_imageio_is_required_when_system_ffmpeg_missing(
        self,
        find_spec_mock,
        which_mock,
    ) -> None:
        which_mock.return_value = None

        def fake_find_spec(module_name: str) -> object | None:
            if module_name in {"torch", "whisper"}:
                return object()
            return None

        find_spec_mock.side_effect = fake_find_spec

        missing_packages = collect_missing_transcription_packages()

        self.assertEqual(["imageio-ffmpeg"], missing_packages)


if __name__ == "__main__":
    unittest.main()
