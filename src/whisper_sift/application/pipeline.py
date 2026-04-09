from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_INTERVIEWER_LABELS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
    DEFAULT_QUESTION_SUFFIX,
    DEFAULT_WRITE_QUESTION_JSON,
    ExtractionPolicy,
    OutputPolicy,
    QuestionExtractionOptions,
    TranscriptionOptions,
)
from whisper_sift.infrastructure.filesystem import write_json_file
from whisper_sift.runtime.reporting import ProgressReporter, report_progress

from .extract_questions import (
    ExtractQuestionsRequest,
    ExtractQuestionsResult,
    run_extract_questions,
)
from .transcribe import TranscribeRequest, TranscribeResult, run_transcribe


@dataclass(slots=True)
class PipelineRequest:
    transcription_options: TranscriptionOptions
    questions_output_dir: Path | None = None
    summary_json_path: Path | None = None
    output: OutputPolicy | None = None
    extraction: ExtractionPolicy | None = None
    suffix: str = DEFAULT_QUESTION_SUFFIX
    write_json: bool = DEFAULT_WRITE_QUESTION_JSON
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS
    reporter: ProgressReporter | None = None

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


@dataclass(slots=True)
class PipelineResult:
    transcription: TranscribeResult
    questions: ExtractQuestionsResult
    summary_json_path: Path | None = None

    @property
    def generated_question_files(self) -> tuple[Path, ...]:
        return self.questions.generated_files

    @property
    def generated_question_json_files(self) -> tuple[Path, ...]:
        return self.questions.generated_json_files

    def to_dict(self) -> dict[str, object]:
        return {
            "transcription": self.transcription.to_dict(),
            "questions": self.questions.to_dict(),
            "question_source_count": len(self.questions.artifacts),
            "generated_question_files": [
                str(path) for path in self.generated_question_files
            ],
            "generated_question_json_files": [
                str(path) for path in self.generated_question_json_files
            ],
        }


def run_pipeline(request: PipelineRequest) -> PipelineResult:
    transcription_result = run_transcribe(
        TranscribeRequest(
            options=request.transcription_options,
            summary_json_path=None,
            reporter=request.reporter,
        )
    )
    transcript_files = _select_question_sources(transcription_result.generated_files)
    question_options = QuestionExtractionOptions(
        files=transcript_files,
        output_dir=request.questions_output_dir,
        output=request.output,
        extraction=request.extraction,
    )
    questions_result = run_extract_questions(
        ExtractQuestionsRequest(
            options=question_options,
            reporter=request.reporter,
        )
    )
    summary_json_path = (
        request.summary_json_path.expanduser().resolve()
        if request.summary_json_path is not None
        else None
    )
    result = PipelineResult(
        transcription=transcription_result,
        questions=questions_result,
        summary_json_path=summary_json_path,
    )
    if summary_json_path is not None:
        write_json_file(summary_json_path, result.to_dict())
        report_progress(
            request.reporter,
            f"[summary] {summary_json_path}",
        )
    return result


def _select_question_sources(generated_files: tuple[Path, ...]) -> list[Path]:
    selected_by_stem: dict[str, Path] = {}
    ordered_stems: list[str] = []

    for path in generated_files:
        suffix = path.suffix.lower()
        if suffix not in {".txt", ".srt"}:
            continue

        stem = path.stem
        if stem not in selected_by_stem:
            ordered_stems.append(stem)
            selected_by_stem[stem] = path
            continue

        if _question_source_priority(suffix) > _question_source_priority(
            selected_by_stem[stem].suffix.lower()
        ):
            selected_by_stem[stem] = path

    return [selected_by_stem[stem] for stem in ordered_stems]


def _question_source_priority(suffix: str) -> int:
    if suffix == ".srt":
        return 2
    if suffix == ".txt":
        return 1
    return 0
