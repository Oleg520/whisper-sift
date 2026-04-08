from __future__ import annotations

import io
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.cli import (
    EXIT_FILE_NOT_FOUND,
    _normalize_output_formats,
    _normalize_pipeline_formats,
    main,
)


class CliTests(unittest.TestCase):
    def test_normalize_output_formats_deduplicates_and_lowercases(self) -> None:
        normalized = _normalize_output_formats(["TXT", "srt", "txt", " SRT "])

        self.assertEqual(("txt", "srt"), normalized)

    def test_normalize_pipeline_formats_adds_txt_when_missing(self) -> None:
        normalized, txt_added = _normalize_pipeline_formats(["srt"])

        self.assertEqual(("srt", "txt"), normalized)
        self.assertTrue(txt_added)

    def test_main_returns_friendly_missing_file_error(self) -> None:
        missing_file = PROJECT_ROOT / "missing_transcript.txt"
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(["extract-questions", str(missing_file)])

        self.assertEqual(EXIT_FILE_NOT_FOUND, exit_code)
        self.assertIn("[error]", stderr.getvalue())
        self.assertIn("Transcript file not found", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
