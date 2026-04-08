from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.utils.text import extract_question_candidates, normalize_whitespace


class TextUtilityTests(unittest.TestCase):
    def test_normalize_whitespace_collapses_spaces_and_newlines(self) -> None:
        value = "  one   two\r\n\r\n three \t four \r five  "

        normalized = normalize_whitespace(value)

        self.assertEqual("one two\n\nthree four\nfive", normalized)

    def test_extract_question_candidates_deduplicates_by_default(self) -> None:
        transcript = (
            "Расскажите, пожалуйста, о вашем опыте работы?\n"
            "Да, конечно.\n"
            "Какие технологии вы использовали на последнем проекте?\n"
            "Java и Spring.\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Расскажите, пожалуйста, о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_keeps_duplicates_when_requested(self) -> None:
        transcript = (
            "Какие технологии вы использовали на последнем проекте?\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript, deduplicate=False)

        self.assertEqual(
            [
                "Какие технологии вы использовали на последнем проекте?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_filters_noise(self) -> None:
        transcript = (
            "test test test test test test test test?\n"
            "Чем вы занимались на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Чем вы занимались на последнем проекте?"],
            questions,
        )


if __name__ == "__main__":
    unittest.main()
