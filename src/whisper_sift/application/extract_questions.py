from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.runtime.reporting import ProgressReporter


@dataclass(slots=True)
class ExtractQuestionsRequest:
    options: QuestionExtractionOptions
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class ExtractQuestionsResult:
    generated_files: tuple[Path, ...]
    generated_json_files: tuple[Path, ...] = ()


def run_extract_questions(request: ExtractQuestionsRequest) -> ExtractQuestionsResult:
    from whisper_sift.services.questions import extract_questions_from_files

    artifacts = extract_questions_from_files(
        request.options,
        reporter=request.reporter,
    )
    return ExtractQuestionsResult(
        generated_files=artifacts.text_files,
        generated_json_files=artifacts.json_files,
    )
