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

    def test_extract_question_candidates_accepts_question_like_phrase_without_mark(self) -> None:
        transcript = (
            "Расскажите, пожалуйста, про ваш последний проект\n"
            "Я работал над платежным модулем.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Расскажите, пожалуйста, про ваш последний проект?"],
            questions,
        )

    def test_extract_question_candidates_accepts_est_li_question_without_mark(self) -> None:
        transcript = (
            "Есть ли у вас опыт работы с Kafka\n"
            "Да, у меня был такой опыт.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Есть ли у вас опыт работы с Kafka?"],
            questions,
        )

    def test_extract_question_candidates_filters_answer_like_phrase_with_question_mark(self) -> None:
        transcript = (
            "Я работал над платежным модулем?\n"
            "Какие технологии вы использовали на последнем проекте?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие технологии вы использовали на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_keeps_question_like_phrase_starting_with_ya(self) -> None:
        transcript = (
            "Я правильно понимаю, что у вас был опыт работы с Kafka\n"
            "Да, был.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Я правильно понимаю, что у вас был опыт работы с Kafka?"],
            questions,
        )

    def test_extract_question_candidates_fuzzy_deduplicates_similar_questions(self) -> None:
        transcript = (
            "Какие технологии вы использовали на последнем проекте?\n"
            "Какие технологии использовали на последнем проекте?\n"
            "Какие технологии вы использовали на последнем проекте ?\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            ["Какие технологии вы использовали на последнем проекте?"],
            questions,
        )

    def test_extract_question_candidates_uses_detected_interviewer_label(self) -> None:
        transcript = (
            "Интервьюер: Расскажите о вашем опыте работы\n"
            "Кандидат: Я работаю в backend уже пять лет.\n"
            "Интервьюер: Какие технологии вы использовали на последнем проекте?\n"
            "Кандидат: Java и Spring.\n"
        )

        questions = extract_question_candidates(transcript)

        self.assertEqual(
            [
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )

    def test_extract_question_candidates_supports_explicit_speaker_label(self) -> None:
        transcript = (
            "SPEAKER_00: Расскажите о вашем опыте работы\n"
            "SPEAKER_01: Я работаю в backend уже пять лет.\n"
            "SPEAKER_00: Какие технологии вы использовали на последнем проекте?\n"
            "SPEAKER_01: Java и Spring.\n"
        )

        questions = extract_question_candidates(
            transcript,
            interviewer_labels=("SPEAKER_00",),
        )

        self.assertEqual(
            [
                "Расскажите о вашем опыте работы?",
                "Какие технологии вы использовали на последнем проекте?",
            ],
            questions,
        )


if __name__ == "__main__":
    unittest.main()
