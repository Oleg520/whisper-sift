from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.domain.evaluation import (
    diff_evaluation_reports,
    evaluate_golden_set,
    load_evaluation_report,
    load_golden_set,
)


class EvaluationTests(unittest.TestCase):
    def test_load_golden_set_resolves_relative_sources(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcripts_dir = workspace / "transcripts"
            transcripts_dir.mkdir()
            transcript_path = transcripts_dir / "sample.txt"
            transcript_path.write_text(
                "Расскажите про ваш проект\n",
                encoding="utf-8",
            )
            manifest_path = workspace / "golden_set.json"
            manifest_path.write_text(
                (
                    "{\n"
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source": "transcripts/sample.txt",\n'
                    '      "required_questions": ["Расскажите про ваш проект?"],\n'
                    '      "forbidden_questions": ["Ну, а что у нас?"],\n'
                    '      "interviewer_labels": ["SPEAKER_00"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            cases = load_golden_set(manifest_path)

            self.assertEqual(1, len(cases))
            self.assertEqual(transcript_path.resolve(), cases[0].source_path)
            self.assertEqual(("Расскажите про ваш проект?",), cases[0].required_questions)
            self.assertEqual(("Ну, а что у нас?",), cases[0].forbidden_questions)
            self.assertEqual(("SPEAKER_00",), cases[0].interviewer_labels)

    def test_evaluate_golden_set_reports_missing_and_forbidden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "sample.txt"
            transcript_path.write_text(
                (
                    "Расскажите про ваш проект\n"
                    "Какие технологии вы использовали?\n"
                ),
                encoding="utf-8",
            )
            manifest_path = workspace / "golden_set.json"
            manifest_path.write_text(
                (
                    "{\n"
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source": "sample.txt",\n'
                    '      "required_questions": ["Расскажите про ваш проект?", "Какой стек использовали?"],\n'
                    '      "forbidden_questions": ["Какие технологии вы использовали?"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            report = evaluate_golden_set(manifest_path)

            self.assertFalse(report.is_passing)
            self.assertEqual(1, report.failed_case_count)
            case = report.cases[0]
            self.assertEqual(("Расскажите про ваш проект?",), case.matched_required)
            self.assertEqual(("Какой стек использовали?",), case.missing_required)
            self.assertEqual(("Какие технологии вы использовали?",), case.present_forbidden)

    def test_evaluate_golden_set_can_filter_by_case_name(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            first_path = workspace / "first.txt"
            second_path = workspace / "second.txt"
            first_path.write_text("Расскажите про ваш проект\n", encoding="utf-8")
            second_path.write_text("Какие технологии использовали?\n", encoding="utf-8")
            manifest_path = workspace / "golden_set.json"
            manifest_path.write_text(
                (
                    "{\n"
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "first",\n'
                    '      "source": "first.txt",\n'
                    '      "required_questions": ["Расскажите про ваш проект?"]\n'
                    "    },\n"
                    "    {\n"
                    '      "name": "second",\n'
                    '      "source": "second.txt",\n'
                    '      "required_questions": ["Какие технологии использовали?"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            report = evaluate_golden_set(manifest_path, selected_cases=("second",))

            self.assertEqual(1, report.case_count)
            self.assertEqual("second", report.cases[0].case.name)

    def test_load_evaluation_report_and_diff_reports(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            baseline_path = workspace / "baseline.json"
            current_path = workspace / "current.json"
            baseline_path.write_text(
                (
                    "{\n"
                    '  "golden_set_path": "golden_set.json",\n'
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source_path": "sample.txt",\n'
                    '      "matched_required": ["Q1"],\n'
                    '      "missing_required": ["Q2"],\n'
                    '      "present_forbidden": ["Bad 1"],\n'
                    '      "extracted_questions": ["Q1", "Bad 1"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            current_path.write_text(
                (
                    "{\n"
                    '  "golden_set_path": "golden_set.json",\n'
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source_path": "sample.txt",\n'
                    '      "matched_required": ["Q1", "Q2"],\n'
                    '      "missing_required": [],\n'
                    '      "present_forbidden": [],\n'
                    '      "extracted_questions": ["Q1", "Q2"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            baseline_report = load_evaluation_report(baseline_path)
            current_report = load_evaluation_report(current_path)
            diff = diff_evaluation_reports(
                baseline_report,
                current_report,
                current_report_path=current_path,
                baseline_report_path=baseline_path,
            )

            self.assertTrue(diff.has_changes)
            self.assertEqual(1, diff.improvement_case_count)
            self.assertEqual(0, diff.regression_case_count)
            case = diff.case_diffs[0]
            self.assertEqual(("Q2",), case.resolved_missing_required)
            self.assertEqual(("Bad 1",), case.resolved_forbidden)
            self.assertEqual(("Q2",), case.added_questions)


if __name__ == "__main__":
    unittest.main()
