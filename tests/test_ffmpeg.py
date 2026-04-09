from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.infrastructure.ffmpeg import ensure_ffmpeg_on_path


class FfmpegTests(unittest.TestCase):
    @patch("whisper_sift.infrastructure.ffmpeg._resolve_imageio_ffmpeg")
    @patch("whisper_sift.infrastructure.ffmpeg._resolve_system_ffmpeg")
    def test_uses_existing_bundled_ffmpeg_without_imageio(
        self,
        resolve_system_mock,
        resolve_imageio_mock,
    ) -> None:
        resolve_system_mock.return_value = None
        resolve_imageio_mock.return_value = None

        with tempfile.TemporaryDirectory() as temp_dir:
            bundled_path = Path(temp_dir) / ("ffmpeg.exe" if os.name == "nt" else "ffmpeg")
            bundled_path.write_text("placeholder", encoding="utf-8")

            with patch.dict(os.environ, {"PATH": ""}, clear=False):
                with patch(
                    "whisper_sift.infrastructure.ffmpeg._bundled_ffmpeg_alias",
                    return_value=bundled_path,
                ):
                    resolved = ensure_ffmpeg_on_path()

        self.assertEqual(bundled_path, resolved)
        self.assertFalse(resolve_imageio_mock.called)


if __name__ == "__main__":
    unittest.main()
