from __future__ import annotations

from whisper_sift.domain.extraction import extract_question_candidates, normalize_whitespace
from whisper_sift.domain.questions import QuestionCandidate, QuestionExtractionResult
from whisper_sift.domain.transcript import (
    SpeakerTurn,
    TranscriptArtifact,
    TranscriptSlice,
    TranscriptView,
)

__all__ = [
    "QuestionCandidate",
    "QuestionExtractionResult",
    "SpeakerTurn",
    "TranscriptArtifact",
    "TranscriptSlice",
    "TranscriptView",
    "extract_question_candidates",
    "normalize_whitespace",
]
