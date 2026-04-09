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
    EXIT_USAGE_ERROR,
    _normalize_argv,
    _normalize_output_formats,
    _normalize_pipeline_formats,
    build_parser,
    main,
)
from whisper_sift.config import (
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
    DEFAULT_OUTPUT_FORMATS,
    DEFAULT_QUESTION_SUFFIX,
    DEFAULT_TRANSCRIPTION_DEVICE,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_WRITE_QUESTION_JSON,
)


class CliTests(unittest.TestCase):
    def test_parser_uses_centralized_defaults(self) -> None:
        parser = build_parser()
        transcribe_args = parser.parse_args(["transcribe", "sample.mkv"])
        extract_args = parser.parse_args(["extract-questions", "sample.txt"])
        evaluate_args = parser.parse_args(["evaluate"])

        self.assertEqual(DEFAULT_TRANSCRIPTION_MODEL, transcribe_args.model)
        self.assertEqual(DEFAULT_TRANSCRIPTION_LANGUAGE, transcribe_args.language)
        self.assertEqual(DEFAULT_TRANSCRIPTION_DEVICE, transcribe_args.device)
        self.assertEqual(list(DEFAULT_OUTPUT_FORMATS), transcribe_args.formats)
        self.assertEqual(DEFAULT_QUESTION_SUFFIX, extract_args.suffix)
        self.assertEqual(DEFAULT_WRITE_QUESTION_JSON, extract_args.json)
        self.assertEqual(DEFAULT_MIN_QUESTION_LENGTH, extract_args.min_length)
        self.assertEqual(DEFAULT_MAX_QUESTION_LENGTH, extract_args.max_length)
        self.assertEqual("golden_set.json", evaluate_args.golden_set.name)
        self.assertEqual([], evaluate_args.case)

    def test_normalize_output_formats_deduplicates_and_lowercases(self) -> None:
        normalized = _normalize_output_formats(["TXT", "srt", "txt", " SRT "])

        self.assertEqual(("txt", "srt"), normalized)

    def test_normalize_pipeline_formats_adds_txt_when_missing(self) -> None:
        normalized, txt_added = _normalize_pipeline_formats(["srt"])

        self.assertEqual(("srt", "txt"), normalized)
        self.assertTrue(txt_added)

    def test_normalize_argv_infers_transcribe_for_file_like_target(self) -> None:
        normalized = _normalize_argv(["interview.mkv", "--output-dir", "results"])

        self.assertEqual(
            ["transcribe", "interview.mkv", "--output-dir", "results"],
            normalized,
        )

    def test_normalize_argv_keeps_unknown_command_like_value(self) -> None:
        normalized = _normalize_argv(["doctro"])

        self.assertEqual(["doctro"], normalized)

    def test_main_returns_friendly_missing_file_error(self) -> None:
        missing_file = PROJECT_ROOT / "missing_transcript.txt"
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(["extract-questions", str(missing_file)])

        self.assertEqual(EXIT_FILE_NOT_FOUND, exit_code)
        self.assertIn("[error]", stderr.getvalue())
        self.assertIn("Transcript file not found", stderr.getvalue())

    def test_main_returns_usage_error_for_unknown_command_typo(self) -> None:
        stderr = io.StringIO()

        with redirect_stderr(stderr):
            exit_code = main(["doctro"])

        self.assertEqual(EXIT_USAGE_ERROR, exit_code)
        self.assertIn("invalid choice", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
