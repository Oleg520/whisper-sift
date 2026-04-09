from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.config import FAKE_TRANSCRIPTION_TEXT_ENV

LAUNCHER = PROJECT_ROOT / "transcribe_whisper.py"


class CliIntegrationTests(unittest.TestCase):
    def test_launcher_transcribe_smoke_with_fixture_backend(self) -> None:
        transcript = (
            "Интервьюер: Расскажите про ваш последний проект\n"
            "Кандидат: Я работал над платежным сервисом.\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            media_path = workspace / "interview.mkv"
            output_dir = workspace / "results"
            summary_path = workspace / "transcribe_summary.json"
            media_path.write_bytes(b"fake media")

            result = self._run_cli(
                str(LAUNCHER),
                "transcribe",
                str(media_path),
                "--output-dir",
                str(output_dir),
                "--formats",
                "txt",
                "--summary-json",
                str(summary_path),
                extra_env={FAKE_TRANSCRIPTION_TEXT_ENV: transcript},
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            output_path = output_dir / "interview.txt"
            self.assertTrue(summary_path.exists())
            self.assertTrue(output_path.exists())
            self.assertEqual(transcript.strip(), output_path.read_text(encoding="utf-8").strip())
            self.assertIn('"generated_file_count": 1', summary_path.read_text(encoding="utf-8"))
            self.assertIn("[backend] fixture transcription backend enabled", result.stdout)
            self.assertIn("[done]", result.stdout)
            self.assertIn("[summary]", result.stdout)

    def test_launcher_extract_questions_smoke(self) -> None:
        transcript = (
            "Интервьюер: Расскажите про ваш последний проект\n"
            "Кандидат: Я работал над платежным сервисом.\n"
            "Интервьюер: Какие технологии вы использовали?\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "interview.txt"
            output_dir = workspace / "questions"
            transcript_path.write_text(transcript, encoding="utf-8")

            result = self._run_cli(
                str(LAUNCHER),
                "extract-questions",
                str(transcript_path),
                "--output-dir",
                str(output_dir),
                "--json",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            output_path = output_dir / "interview_questions.txt"
            json_path = output_dir / "interview_questions.json"
            self.assertTrue(output_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(
                (
                    "Расскажите про ваш последний проект?\n"
                    "Какие технологии вы использовали?"
                ),
                output_path.read_text(encoding="utf-8").strip(),
            )
            self.assertIn('"question_count": 2', json_path.read_text(encoding="utf-8"))
            self.assertIn("[questions]", result.stdout)

    def test_launcher_extract_questions_smoke_from_srt(self) -> None:
        transcript = (
            "1\n"
            "00:00:00,000 --> 00:00:02,000\n"
            "Можешь рассказать\n\n"
            "2\n"
            "00:00:02,000 --> 00:00:05,000\n"
            "про Spring и Spring Boot?\n\n"
            "3\n"
            "00:00:05,000 --> 00:00:06,000\n"
            "Да, конечно.\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "interview.srt"
            output_dir = workspace / "questions"
            transcript_path.write_text(transcript, encoding="utf-8")

            result = self._run_cli(
                str(LAUNCHER),
                "extract-questions",
                str(transcript_path),
                "--output-dir",
                str(output_dir),
                "--json",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            output_path = output_dir / "interview_questions.txt"
            json_path = output_dir / "interview_questions.json"
            self.assertTrue(output_path.exists())
            self.assertTrue(json_path.exists())
            self.assertEqual(
                "Можешь рассказать про Spring и Spring Boot?",
                output_path.read_text(encoding="utf-8").strip(),
            )
            json_text = json_path.read_text(encoding="utf-8")
            self.assertIn('"start_time": "00:00:00,000"', json_text)
            self.assertIn('"end_time": "00:00:05,000"', json_text)
            self.assertIn("[questions]", result.stdout)

    def test_module_pipeline_smoke_with_fixture_backend(self) -> None:
        transcript = (
            "Интервьюер: Расскажите про ваш последний проект\n"
            "Кандидат: Я работал над платежным сервисом.\n"
            "Интервьюер: Какие технологии вы использовали?\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            media_path = workspace / "interview.mkv"
            transcript_dir = workspace / "transcripts"
            questions_dir = workspace / "questions"
            summary_path = workspace / "pipeline_summary.json"
            media_path.write_bytes(b"fake media")

            result = self._run_cli(
                "-m",
                "whisper_sift",
                "pipeline",
                str(media_path),
                "--output-dir",
                str(transcript_dir),
                "--questions-dir",
                str(questions_dir),
                "--formats",
                "txt",
                "--json",
                "--summary-json",
                str(summary_path),
                extra_env={FAKE_TRANSCRIPTION_TEXT_ENV: transcript},
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            transcript_path = transcript_dir / "interview.txt"
            questions_path = questions_dir / "interview_questions.txt"
            questions_json_path = questions_dir / "interview_questions.json"
            self.assertTrue(summary_path.exists())
            self.assertTrue(transcript_path.exists())
            self.assertTrue(questions_path.exists())
            self.assertTrue(questions_json_path.exists())
            self.assertEqual(
                (
                    "Расскажите про ваш последний проект?\n"
                    "Какие технологии вы использовали?"
                ),
                questions_path.read_text(encoding="utf-8").strip(),
            )
            self.assertIn('"question_count": 2', questions_json_path.read_text(encoding="utf-8"))
            summary_text = summary_path.read_text(encoding="utf-8")
            self.assertIn('"generated_question_files"', summary_text)
            self.assertIn('"question_source_count": 1', summary_text)
            self.assertIn("[done]", result.stdout)
            self.assertIn("[questions]", result.stdout)
            self.assertIn("[summary]", result.stdout)

    def test_module_extract_questions_smoke_with_explicit_interviewer_label(self) -> None:
        transcript = (
            "Interviewer: Tell me about your latest project\n"
            "Candidate: I built internal tooling for analytics.\n"
            "Candidate: Can I clarify one thing?\n"
            "Interviewer: What stack did you use?\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "diarized.txt"
            output_dir = workspace / "results"
            transcript_path.write_text(transcript, encoding="utf-8")

            result = self._run_cli(
                "-m",
                "whisper_sift",
                "extract-questions",
                str(transcript_path),
                "--output-dir",
                str(output_dir),
                "--suffix",
                "_only_questions.txt",
                "--interviewer-label",
                "Interviewer",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            output_path = output_dir / "diarized_only_questions.txt"
            self.assertTrue(output_path.exists())
            self.assertEqual(
                (
                    "Tell me about your latest project?\n"
                    "What stack did you use?"
                ),
                output_path.read_text(encoding="utf-8").strip(),
            )
            self.assertIn("[questions]", result.stdout)

    def test_launcher_evaluate_smoke(self) -> None:
        transcript = (
            "Расскажите про ваш последний проект\n"
            "Какие технологии вы использовали?\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "sample.txt"
            golden_set_path = workspace / "golden_set.json"
            report_path = workspace / "latest_report.json"
            transcript_path.write_text(transcript, encoding="utf-8")
            golden_set_path.write_text(
                (
                    "{\n"
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source": "sample.txt",\n'
                    '      "required_questions": [\n'
                    '        "Расскажите про ваш последний проект?",\n'
                    '        "Какие технологии вы использовали?"\n'
                    "      ],\n"
                    '      "forbidden_questions": ["Ну, а что у нас?"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            result = self._run_cli(
                str(LAUNCHER),
                "evaluate",
                "--golden-set",
                str(golden_set_path),
                "--report-json",
                str(report_path),
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            self.assertTrue(report_path.exists())
            self.assertIn('"is_passing": true', report_path.read_text(encoding="utf-8").lower())
            self.assertIn("[eval] sample: PASS", result.stdout)

    def test_launcher_evaluate_can_compare_with_baseline_and_update_it(self) -> None:
        transcript = (
            "Расскажите про ваш последний проект\n"
            "Какие технологии вы использовали?\n"
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            transcript_path = workspace / "sample.txt"
            golden_set_path = workspace / "golden_set.json"
            report_path = workspace / "latest_report.json"
            baseline_path = workspace / "baseline_report.json"
            diff_path = workspace / "latest_diff.json"
            transcript_path.write_text(transcript, encoding="utf-8")
            golden_set_path.write_text(
                (
                    "{\n"
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source": "sample.txt",\n'
                    '      "required_questions": [\n'
                    '        "Расскажите про ваш последний проект?",\n'
                    '        "Какие технологии вы использовали?"\n'
                    "      ],\n"
                    '      "forbidden_questions": []\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )
            baseline_path.write_text(
                (
                    "{\n"
                    '  "golden_set_path": "golden_set.json",\n'
                    '  "cases": [\n'
                    "    {\n"
                    '      "name": "sample",\n'
                    '      "source_path": "sample.txt",\n'
                    '      "matched_required": ["Расскажите про ваш последний проект?"],\n'
                    '      "missing_required": ["Какие технологии вы использовали?"],\n'
                    '      "present_forbidden": [],\n'
                    '      "extracted_questions": ["Расскажите про ваш последний проект?"]\n'
                    "    }\n"
                    "  ]\n"
                    "}\n"
                ),
                encoding="utf-8",
            )

            result = self._run_cli(
                str(LAUNCHER),
                "evaluate",
                "--golden-set",
                str(golden_set_path),
                "--report-json",
                str(report_path),
                "--baseline-report",
                str(baseline_path),
                "--diff-json",
                str(diff_path),
                "--update-baseline",
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            self.assertTrue(diff_path.exists())
            self.assertIn('"improvement_case_count": 1', diff_path.read_text(encoding="utf-8"))
            self.assertIn('"is_passing": true', baseline_path.read_text(encoding="utf-8").lower())
            self.assertIn("[eval:diff] sample: improvement", result.stdout)
            self.assertIn("[eval] Baseline updated", result.stdout)

    def _run_cli(
        self,
        *arguments: str,
        extra_env: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        pythonpath_parts = [str(SRC_DIR)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)
        if extra_env:
            env.update(extra_env)

        return subprocess.run(
            [sys.executable, *arguments],
            cwd=PROJECT_ROOT,
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
        )


if __name__ == "__main__":
    unittest.main()
