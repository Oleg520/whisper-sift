from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.domain.evaluation import evaluate_golden_set, load_golden_set


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


if __name__ == "__main__":
    unittest.main()
