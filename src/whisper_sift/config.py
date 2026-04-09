from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_DIR = Path(".")
DEFAULT_TRANSCRIPTION_MODEL = "small"
DEFAULT_TRANSCRIPTION_LANGUAGE = "ru"
AUTO_DETECT_LANGUAGE = "auto"
DEFAULT_TRANSCRIPTION_DEVICE = "auto"
DEFAULT_OUTPUT_FORMATS = ("txt", "srt")
DEFAULT_QUESTION_SUFFIX = "_questions.txt"
DEFAULT_DEDUPLICATE_QUESTIONS = True
DEFAULT_MIN_QUESTION_LENGTH = 10
DEFAULT_MAX_QUESTION_LENGTH = 240
DEFAULT_INTERVIEWER_LABELS: tuple[str, ...] = ()


def normalize_transcription_language(value: str | None) -> str | None:
    if value is None:
        return None

    normalized = value.strip()
    if not normalized:
        return None
    if normalized.lower() == AUTO_DETECT_LANGUAGE:
        return None
    return normalized


@dataclass(slots=True)
class TranscriptionOptions:
    files: list[Path]
    output_dir: Path
    model: str = DEFAULT_TRANSCRIPTION_MODEL
    language: str | None = DEFAULT_TRANSCRIPTION_LANGUAGE
    device: str = DEFAULT_TRANSCRIPTION_DEVICE
    formats: tuple[str, ...] = DEFAULT_OUTPUT_FORMATS


@dataclass(slots=True)
class QuestionExtractionOptions:
    files: list[Path]
    output_dir: Path | None = None
    suffix: str = DEFAULT_QUESTION_SUFFIX
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS
