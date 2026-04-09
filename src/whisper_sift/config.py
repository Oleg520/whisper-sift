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
DEFAULT_WRITE_QUESTION_JSON = False
DEFAULT_DEDUPLICATE_QUESTIONS = True
DEFAULT_MIN_QUESTION_LENGTH = 10
DEFAULT_MAX_QUESTION_LENGTH = 240
DEFAULT_INTERVIEWER_LABELS: tuple[str, ...] = ()
FAKE_TRANSCRIPTION_TEXT_ENV = "WHISPER_SIFT_FAKE_TRANSCRIPTION_TEXT"
FAKE_TRANSCRIPTION_FILE_ENV = "WHISPER_SIFT_FAKE_TRANSCRIPTION_FILE"


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
class OutputPolicy:
    suffix: str = DEFAULT_QUESTION_SUFFIX
    write_json: bool = DEFAULT_WRITE_QUESTION_JSON


@dataclass(slots=True)
class ExtractionPolicy:
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS


@dataclass(slots=True)
class EvaluationPolicy:
    selected_cases: tuple[str, ...] = ()
    update_baseline: bool = False


@dataclass(slots=True)
class QuestionExtractionOptions:
    files: list[Path]
    output_dir: Path | None = None
    output: OutputPolicy | None = None
    extraction: ExtractionPolicy | None = None
    suffix: str = DEFAULT_QUESTION_SUFFIX
    write_json: bool = DEFAULT_WRITE_QUESTION_JSON
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS

    def __post_init__(self) -> None:
        if self.output is None:
            self.output = OutputPolicy(
                suffix=self.suffix,
                write_json=self.write_json,
            )
        else:
            self.suffix = self.output.suffix
            self.write_json = self.output.write_json

        if self.extraction is None:
            self.extraction = ExtractionPolicy(
                deduplicate=self.deduplicate,
                min_length=self.min_length,
                max_length=self.max_length,
                interviewer_labels=self.interviewer_labels,
            )
        else:
            self.deduplicate = self.extraction.deduplicate
            self.min_length = self.extraction.min_length
            self.max_length = self.extraction.max_length
            self.interviewer_labels = self.extraction.interviewer_labels
