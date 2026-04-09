from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
LAUNCHER = PROJECT_ROOT / "transcribe_whisper.py"


class CliIntegrationTests(unittest.TestCase):
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
            )

            self.assertEqual(0, result.returncode, msg=result.stderr or result.stdout)
            output_path = output_dir / "interview_questions.txt"
            self.assertTrue(output_path.exists())
            self.assertEqual(
                (
                    "Расскажите про ваш последний проект?\n"
                    "Какие технологии вы использовали?"
                ),
                output_path.read_text(encoding="utf-8").strip(),
            )
            self.assertIn("[questions]", result.stdout)

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

    def _run_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        pythonpath_parts = [str(SRC_DIR)]
        existing_pythonpath = env.get("PYTHONPATH")
        if existing_pythonpath:
            pythonpath_parts.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

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
