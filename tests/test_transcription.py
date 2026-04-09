from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.config import TranscriptionOptions
from whisper_sift.services.transcription import transcribe_files


class TranscriptionTests(unittest.TestCase):
    @patch("whisper_sift.services.transcription.ensure_ffmpeg_on_path")
    @patch("whisper_sift.services.transcription.whisper.load_model")
    def test_missing_input_fails_before_runtime_setup(
        self,
        load_model_mock,
        ensure_ffmpeg_mock,
    ) -> None:
        missing_file = PROJECT_ROOT / "missing_audio.mkv"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"

            with self.assertRaises(FileNotFoundError):
                transcribe_files(
                    TranscriptionOptions(
                        files=[missing_file],
                        output_dir=output_dir,
                    )
                )

        load_model_mock.assert_not_called()
        ensure_ffmpeg_mock.assert_not_called()
        self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
