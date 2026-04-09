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

    def test_extract_questions_can_be_serialized_to_dict(self) -> None:
        transcript = (
            "Interviewer: Tell me about your latest project\n"
            "Candidate: I built internal tooling.\n"
        )

        result = extract_questions(transcript, source_name="sample.txt")
        payload = result.to_dict(source_path="/tmp/sample.txt")

        self.assertEqual("sample.txt", payload["source_name"])
        self.assertEqual("/tmp/sample.txt", payload["source_path"])
        self.assertEqual(1, payload["question_count"])
        self.assertEqual(
            ["Tell me about your latest project?"],
            payload["question_texts"],
        )
        self.assertIn("transcript", payload)
        self.assertIn("questions", payload)

    def test_extract_questions_keeps_raw_source_text_alongside_normalized_output(self) -> None:
        transcript = "Кстати, поджаве, вы-то получается, в основном, в 17-е, да, работал?\n"

        result = extract_questions(transcript)

        self.assertEqual(
            "По Java вы в основном с 17-й версией работали?",
            result.questions[0].text,
        )
        self.assertEqual(
            "Кстати, поджаве, вы-то получается, в основном, в 17-е, да, работал",
            result.questions[0].source_text,
        )

    def test_extract_questions_does_not_treat_dash_inside_words_as_speaker_split(self) -> None:
        transcript = (
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "А вообще лист сет Q-мэп, что это такое? Это интерфейс или класс?\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:05,000\n"
            "Это интерфейс.\n"
        )

        result = extract_questions(transcript, source_name="sample.srt")

        self.assertFalse(result.transcript.has_speaker_structure)
        self.assertTrue(any("что это такое" in text.lower() for text in result.question_texts))

    def test_extract_questions_does_not_treat_question_phrase_as_speaker_label(self) -> None:
        transcript = (
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "И еще один вопрос. Приходился ли когда-нибудь смотреть Java дамп и thread dump?\n\n"
            "2\n"
            "00:00:03,000 --> 00:00:06,000\n"
            "Нет, такого опыта не было.\n"
        )

        result = extract_questions(transcript, source_name="sample.srt")

        self.assertFalse(result.transcript.has_speaker_structure)
        self.assertTrue(any("thread dump" in text.lower() for text in result.question_texts))


if __name__ == "__main__":
    unittest.main()
