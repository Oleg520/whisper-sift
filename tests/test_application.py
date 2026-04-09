from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.application.doctor import DoctorRequest, run_doctor
from whisper_sift.application.extract_questions import (
    ExtractQuestionsRequest,
    run_extract_questions,
)
from whisper_sift.application.pipeline import PipelineRequest, run_pipeline
from whisper_sift.application.transcribe import TranscribeRequest, run_transcribe
from whisper_sift.config import (
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
    DEFAULT_QUESTION_SUFFIX,
    QuestionExtractionOptions,
    TranscriptionOptions,
)


class ApplicationTests(unittest.TestCase):
    @patch("whisper_sift.services.transcription.transcribe_files")
    @patch("whisper_sift.runtime.dependencies.ensure_transcription_dependencies")
    def test_run_transcribe_bootstraps_and_returns_result(
        self,
        ensure_dependencies_mock,
        transcribe_files_mock,
    ) -> None:
        options = TranscriptionOptions(
            files=[Path("interview.mkv")],
            output_dir=Path("results"),
        )
        transcribe_files_mock.return_value = [Path("results/interview.txt")]

        result = run_transcribe(TranscribeRequest(options=options))

        ensure_dependencies_mock.assert_called_once_with()
        transcribe_files_mock.assert_called_once_with(options, reporter=None)
        self.assertEqual((Path("results/interview.txt"),), result.generated_files)

    @patch("whisper_sift.services.questions.extract_questions_from_files")
    def test_run_extract_questions_returns_result(self, extract_questions_mock) -> None:
        options = QuestionExtractionOptions(files=[Path("interview.txt")])
        extract_questions_mock.return_value = [Path("interview_questions.txt")]

        result = run_extract_questions(ExtractQuestionsRequest(options=options))

        extract_questions_mock.assert_called_once_with(options, reporter=None)
        self.assertEqual((Path("interview_questions.txt"),), result.generated_files)

    @patch("whisper_sift.application.pipeline.run_extract_questions")
    @patch("whisper_sift.application.pipeline.run_transcribe")
    def test_run_pipeline_chains_transcribe_and_question_extraction(
        self,
        run_transcribe_mock,
        run_extract_questions_mock,
    ) -> None:
        from whisper_sift.application.extract_questions import ExtractQuestionsResult
        from whisper_sift.application.transcribe import TranscribeResult

        transcription_options = TranscriptionOptions(
            files=[Path("interview.mkv")],
            output_dir=Path("results"),
            formats=("txt", "srt"),
        )
        run_transcribe_mock.return_value = TranscribeResult(
            generated_files=(Path("results/interview.txt"), Path("results/interview.srt"))
        )
        run_extract_questions_mock.return_value = ExtractQuestionsResult(
            generated_files=(Path("questions/interview_questions.txt"),)
        )

        result = run_pipeline(
            PipelineRequest(
                transcription_options=transcription_options,
                questions_output_dir=Path("questions"),
                suffix=DEFAULT_QUESTION_SUFFIX,
                deduplicate=True,
                min_length=DEFAULT_MIN_QUESTION_LENGTH,
                max_length=DEFAULT_MAX_QUESTION_LENGTH,
                interviewer_labels=("SPEAKER_00",),
            )
        )

        run_transcribe_mock.assert_called_once()
        run_extract_questions_mock.assert_called_once()
        extraction_request = run_extract_questions_mock.call_args.args[0]
        self.assertEqual([Path("results/interview.txt")], extraction_request.options.files)
        self.assertEqual(Path("questions"), extraction_request.options.output_dir)
        self.assertEqual(("SPEAKER_00",), extraction_request.options.interviewer_labels)
        self.assertEqual(
            (Path("questions/interview_questions.txt"),),
            result.generated_question_files,
        )

    @patch("whisper_sift.runtime.doctor.collect_doctor_report")
    def test_run_doctor_wraps_report(self, collect_doctor_report_mock) -> None:
        sentinel_report = object()
        collect_doctor_report_mock.return_value = sentinel_report

        result = run_doctor(DoctorRequest(install_missing=True))

        collect_doctor_report_mock.assert_called_once_with(install_missing=True)
        self.assertIs(sentinel_report, result.report)


if __name__ == "__main__":
    unittest.main()
