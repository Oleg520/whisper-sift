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

from whisper_sift.application.doctor import DoctorRequest, run_doctor
from whisper_sift.application.evaluate import EvaluateRequest, run_evaluate
from whisper_sift.application.extract_questions import (
    ExtractQuestionsRequest,
    run_extract_questions,
)
from whisper_sift.application.pipeline import PipelineRequest, run_pipeline
from whisper_sift.application.provision_runtime import (
    ProvisionTranscriptionRuntimeRequest,
    run_provision_transcription_runtime,
)
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
    def test_run_transcribe_returns_result(
        self,
        transcribe_files_mock,
    ) -> None:
        options = TranscriptionOptions(
            files=[Path("interview.mkv")],
            output_dir=Path("results"),
        )
        transcribe_files_mock.return_value = [Path("results/interview.txt")]

        result = run_transcribe(TranscribeRequest(options=options))

        transcribe_files_mock.assert_called_once_with(options, reporter=None)
        self.assertEqual((Path("results/interview.txt"),), result.generated_files)

    @patch("whisper_sift.runtime.dependencies.ensure_transcription_dependencies")
    def test_run_provision_transcription_runtime_bootstraps_dependencies(
        self,
        ensure_dependencies_mock,
    ) -> None:
        result = run_provision_transcription_runtime(
            ProvisionTranscriptionRuntimeRequest()
        )

        ensure_dependencies_mock.assert_called_once_with()
        self.assertTrue(result.ready)

    @patch("whisper_sift.services.questions.extract_questions_from_files")
    def test_run_extract_questions_returns_result(self, extract_questions_mock) -> None:
        options = QuestionExtractionOptions(files=[Path("interview.txt")])
        from whisper_sift.services.questions import QuestionOutputArtifacts

        extract_questions_mock.return_value = QuestionOutputArtifacts(
            text_files=(Path("interview_questions.txt"),),
            json_files=(Path("interview_questions.json"),),
        )

        result = run_extract_questions(ExtractQuestionsRequest(options=options))

        extract_questions_mock.assert_called_once_with(options, reporter=None)
        self.assertEqual((Path("interview_questions.txt"),), result.generated_files)
        self.assertEqual((Path("interview_questions.json"),), result.generated_json_files)

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
            generated_files=(Path("questions/interview_questions.txt"),),
            generated_json_files=(Path("questions/interview_questions.json"),),
        )

        result = run_pipeline(
            PipelineRequest(
                transcription_options=transcription_options,
                questions_output_dir=Path("questions"),
                suffix=DEFAULT_QUESTION_SUFFIX,
                write_json=True,
                deduplicate=True,
                min_length=DEFAULT_MIN_QUESTION_LENGTH,
                max_length=DEFAULT_MAX_QUESTION_LENGTH,
                interviewer_labels=("SPEAKER_00",),
            )
        )

        run_transcribe_mock.assert_called_once()
        run_extract_questions_mock.assert_called_once()
        extraction_request = run_extract_questions_mock.call_args.args[0]
        self.assertEqual([Path("results/interview.srt")], extraction_request.options.files)
        self.assertEqual(Path("questions"), extraction_request.options.output_dir)
        self.assertTrue(extraction_request.options.write_json)
        self.assertEqual(("SPEAKER_00",), extraction_request.options.interviewer_labels)
        self.assertEqual(
            (Path("questions/interview_questions.txt"),),
            result.generated_question_files,
        )
        self.assertEqual(
            (Path("questions/interview_questions.json"),),
            result.generated_question_json_files,
        )

    @patch("whisper_sift.runtime.doctor.collect_doctor_report")
    def test_run_doctor_wraps_report(self, collect_doctor_report_mock) -> None:
        sentinel_report = object()
        collect_doctor_report_mock.return_value = sentinel_report

        result = run_doctor(DoctorRequest(install_missing=True))

        collect_doctor_report_mock.assert_called_once_with(install_missing=True)
        self.assertIs(sentinel_report, result.report)

    @patch("whisper_sift.application.evaluate.write_evaluation_report")
    @patch("whisper_sift.application.evaluate.evaluate_cases")
    @patch("whisper_sift.application.evaluate.load_golden_set")
    def test_run_evaluate_wraps_report_and_writes_json(
        self,
        load_golden_set_mock,
        evaluate_cases_mock,
        write_report_mock,
    ) -> None:
        class _FakeReport:
            is_passing = True
            cases = ()
            passed_case_count = 1
            case_count = 1
            required_matched = 2
            required_total = 2
            forbidden_present = 0
            forbidden_total = 1

            def to_dict(self) -> dict[str, object]:
                return {"ok": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_set = workspace / "golden_set.json"
            report_json = workspace / "latest_report.json"
            golden_set.write_text('{"cases": []}', encoding="utf-8")
            fake_cases = (object(),)
            load_golden_set_mock.return_value = fake_cases
            evaluate_cases_mock.return_value = _FakeReport()
            def _write_report(report, path):
                path.write_text('{"ok": true}', encoding="utf-8")
                return path

            write_report_mock.side_effect = _write_report

            result = run_evaluate(
                EvaluateRequest(
                    golden_set_path=golden_set,
                    report_json_path=report_json,
                )
            )

            load_golden_set_mock.assert_called_once_with(
                golden_set,
                selected_cases=(),
            )
            evaluate_cases_mock.assert_called_once_with(
                fake_cases,
                golden_set_path=golden_set,
            )
            self.assertEqual(report_json, result.report_json_path)
            self.assertIn('"ok": true', report_json.read_text(encoding="utf-8").lower())

    @patch("whisper_sift.application.evaluate.write_evaluation_report")
    @patch("whisper_sift.application.evaluate.write_evaluation_diff")
    @patch("whisper_sift.application.evaluate.load_evaluation_report")
    @patch("whisper_sift.application.evaluate.diff_evaluation_reports")
    @patch("whisper_sift.application.evaluate.evaluate_cases")
    @patch("whisper_sift.application.evaluate.load_golden_set")
    def test_run_evaluate_can_update_baseline_and_write_diff(
        self,
        load_golden_set_mock,
        evaluate_cases_mock,
        diff_reports_mock,
        load_report_mock,
        write_diff_mock,
        write_report_mock,
    ) -> None:
        class _FakeReport:
            is_passing = True
            cases = ()
            passed_case_count = 1
            case_count = 1
            required_matched = 2
            required_total = 2
            forbidden_present = 0
            forbidden_total = 1

            def to_dict(self) -> dict[str, object]:
                return {"ok": True}

        class _FakeDiff:
            changed_case_count = 1
            regression_case_count = 0
            improvement_case_count = 1
            case_diffs = ()

            def to_dict(self) -> dict[str, object]:
                return {"diff": True}

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            golden_set = workspace / "golden_set.json"
            report_json = workspace / "latest_report.json"
            baseline_json = workspace / "baseline_report.json"
            diff_json = workspace / "latest_diff.json"
            golden_set.write_text('{"cases": []}', encoding="utf-8")
            baseline_json.write_text('{"cases": []}', encoding="utf-8")
            fake_cases = (object(),)
            load_golden_set_mock.return_value = fake_cases
            evaluate_cases_mock.return_value = _FakeReport()
            load_report_mock.return_value = _FakeReport()
            diff_reports_mock.return_value = _FakeDiff()
            def _write_report(report, path):
                path.write_text('{"ok": true}', encoding="utf-8")
                return path

            def _write_diff(diff, path):
                path.write_text('{"diff": true}', encoding="utf-8")
                return path

            write_report_mock.side_effect = _write_report
            write_diff_mock.side_effect = _write_diff

            result = run_evaluate(
                EvaluateRequest(
                    golden_set_path=golden_set,
                    report_json_path=report_json,
                    baseline_report_path=baseline_json,
                    diff_json_path=diff_json,
                    update_baseline=True,
                )
            )

            self.assertEqual(diff_json, result.diff_json_path)
            self.assertEqual(baseline_json, result.baseline_report_path)
            self.assertIn('"diff": true', diff_json.read_text(encoding="utf-8").lower())
            self.assertIn('"ok": true', baseline_json.read_text(encoding="utf-8").lower())


if __name__ == "__main__":
    unittest.main()
