from __future__ import annotations

from dataclasses import dataclass

from whisper_sift.domain.transcript import TranscriptView


@dataclass(slots=True, frozen=True)
class QuestionCandidate:
    text: str
    source_text: str
    explicit_question: bool
    speaker_label: str | None = None
    normalized_speaker_label: str | None = None


@dataclass(slots=True, frozen=True)
class QuestionExtractionResult:
    transcript: TranscriptView
    questions: tuple[QuestionCandidate, ...]
    deduplicated: bool
    min_length: int
    max_length: int

    @property
    def question_texts(self) -> tuple[str, ...]:
        return tuple(candidate.text for candidate in self.questions)
