from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_INTERVIEWER_LABELS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
    DEFAULT_QUESTION_SUFFIX,
    QuestionExtractionOptions,
    TranscriptionOptions,
)
from whisper_sift.runtime.reporting import ProgressReporter

from .extract_questions import ExtractQuestionsRequest, run_extract_questions
from .transcribe import TranscribeRequest, TranscribeResult, run_transcribe


@dataclass(slots=True)
class PipelineRequest:
    transcription_options: TranscriptionOptions
    questions_output_dir: Path | None = None
    suffix: str = DEFAULT_QUESTION_SUFFIX
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class PipelineResult:
    transcription: TranscribeResult
    generated_question_files: tuple[Path, ...]


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    transcription_result = run_transcribe(
        TranscribeRequest(
            options=request.transcription_options,
            reporter=request.reporter,
        )
    )
    transcript_files = [
        path for path in transcription_result.generated_files if path.suffix.lower() == ".txt"
    ]
    question_options = QuestionExtractionOptions(
        files=transcript_files,
        output_dir=request.questions_output_dir,
        suffix=request.suffix,
        deduplicate=request.deduplicate,
        min_length=request.min_length,
        max_length=request.max_length,
        interviewer_labels=request.interviewer_labels,
    )
    questions_result = run_extract_questions(
        ExtractQuestionsRequest(
            options=question_options,
            reporter=request.reporter,
        )
    )
    return PipelineResult(
        transcription=transcription_result,
        generated_question_files=questions_result.generated_files,
    )
