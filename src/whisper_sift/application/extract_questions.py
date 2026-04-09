from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.domain.questions import QuestionOutputArtifact
from whisper_sift.runtime.reporting import ProgressReporter


@dataclass(slots=True)
class ExtractQuestionsRequest:
    options: QuestionExtractionOptions
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class ExtractQuestionsResult:
    artifacts: tuple[QuestionOutputArtifact, ...]

    @property
    def generated_files(self) -> tuple[Path, ...]:
        return tuple(artifact.text_file for artifact in self.artifacts)

    @property
    def generated_json_files(self) -> tuple[Path, ...]:
        return tuple(
            artifact.json_file
            for artifact in self.artifacts
            if artifact.json_file is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": len(self.artifacts),
            "generated_files": [str(path) for path in self.generated_files],
            "generated_json_files": [str(path) for path in self.generated_json_files],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def run_extract_questions(request: ExtractQuestionsRequest) -> ExtractQuestionsResult:
    from whisper_sift.services.questions import extract_questions_from_files

    artifacts = extract_questions_from_files(
        request.options,
        reporter=request.reporter,
    )
    return ExtractQuestionsResult(
        artifacts=artifacts.items,
    )
