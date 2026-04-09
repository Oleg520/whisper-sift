from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import TranscriptionOptions
from whisper_sift.domain.transcription import TranscriptionArtifact
from whisper_sift.infrastructure.filesystem import write_json_file
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


@dataclass(slots=True)
class TranscribeRequest:
    options: TranscriptionOptions
    summary_json_path: Path | None = None
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class TranscribeResult:
    artifacts: tuple[TranscriptionArtifact, ...]
    summary_json_path: Path | None = None

    @property
    def generated_files(self) -> tuple[Path, ...]:
        generated_files: list[Path] = []
        for artifact in self.artifacts:
            generated_files.extend(artifact.generated_files)
        return tuple(generated_files)

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": len(self.artifacts),
            "generated_file_count": len(self.generated_files),
            "generated_files": [str(path) for path in self.generated_files],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }


def run_transcribe(request: TranscribeRequest) -> TranscribeResult:
    from whisper_sift.services.transcription import transcribe_sources

    batch_result = transcribe_sources(request.options, reporter=request.reporter)
    summary_json_path = (
        request.summary_json_path.expanduser().resolve()
        if request.summary_json_path is not None
        else None
    )
    result = TranscribeResult(
        artifacts=batch_result.artifacts,
        summary_json_path=summary_json_path,
    )
    if summary_json_path is not None:
        write_json_file(summary_json_path, result.to_dict())
        report_progress(
            request.reporter,
            f"[summary] {summary_json_path}",
        )
    return result
