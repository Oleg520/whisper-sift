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

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "source_text": self.source_text,
            "explicit_question": self.explicit_question,
            "speaker_label": self.speaker_label,
            "normalized_speaker_label": self.normalized_speaker_label,
        }


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

    def to_dict(self, *, source_path: str | None = None) -> dict[str, object]:
        return {
            "source_name": self.transcript.artifact.source_name,
            "source_path": source_path,
            "deduplicated": self.deduplicated,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "question_count": len(self.questions),
            "question_texts": list(self.question_texts),
            "questions": [candidate.to_dict() for candidate in self.questions],
            "transcript": self.transcript.to_dict(),
        }
