from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.domain.extraction import extract_questions


class DomainExtractionTests(unittest.TestCase):
    def test_extract_questions_returns_structured_result(self) -> None:
        transcript = (
            "Интервьюер: Расскажите о вашем опыте работы\n"
            "Кандидат: Я работаю в backend уже пять лет.\n"
            "Интервьюер: Какие технологии вы использовали на последнем проекте?\n"
            "Кандидат: Python и FastAPI.\n"
        )

        result = extract_questions(transcript, source_name="interview.txt")

        self.assertEqual("interview.txt", result.transcript.artifact.source_name)
        self.assertTrue(result.transcript.has_speaker_structure)
        self.assertEqual(("интервьюер",), result.transcript.selected_interviewer_labels)
        self.assertEqual(
            (
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ),
            result.question_texts,
        )
        self.assertEqual("Интервьюер", result.questions[0].speaker_label)

    def test_extract_questions_keeps_single_raw_slice_without_speakers(self) -> None:
        transcript = (
            "Расскажите, пожалуйста, про ваш последний проект\n"
            "Я работал над платежным модулем.\n"
        )

        result = extract_questions(transcript)

        self.assertFalse(result.transcript.has_speaker_structure)
        self.assertEqual(1, len(result.transcript.candidate_slices))
        self.assertEqual(
            ("Расскажите, пожалуйста, про ваш последний проект?",),
            result.question_texts,
        )


if __name__ == "__main__":
    unittest.main()
