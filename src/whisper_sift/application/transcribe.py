from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import TranscriptionOptions
from whisper_sift.domain.transcription import TranscriptionArtifact
from whisper_sift.runtime.reporting import ProgressReporter


@dataclass(slots=True)
class TranscribeRequest:
    options: TranscriptionOptions
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class TranscribeResult:
    artifacts: tuple[TranscriptionArtifact, ...]

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
    return TranscribeResult(artifacts=batch_result.artifacts)
