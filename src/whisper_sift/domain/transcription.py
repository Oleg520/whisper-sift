from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(slots=True, frozen=True)
class TranscriptionDocument:
    text: str
    language: str | None = None
    segments: tuple[TranscriptionSegment, ...] = ()

    @property
    def segment_count(self) -> int:
        return len(self.segments)


@dataclass(slots=True, frozen=True)
class TranscriptionOutputFile:
    path: Path
    output_format: str

    def to_dict(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "output_format": self.output_format,
        }


@dataclass(slots=True, frozen=True)
class TranscriptionArtifact:
    source_path: Path
    document: TranscriptionDocument
    outputs: tuple[TranscriptionOutputFile, ...]
    model_name: str
    requested_device: str
    resolved_device: str
    use_fp16: bool

    @property
    def source_name(self) -> str:
        return self.source_path.name

    @property
    def generated_files(self) -> tuple[Path, ...]:
        return tuple(output.path for output in self.outputs)

    def to_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "source_path": str(self.source_path),
            "model_name": self.model_name,
            "requested_device": self.requested_device,
            "resolved_device": self.resolved_device,
            "use_fp16": self.use_fp16,
            "language": self.document.language,
            "segment_count": self.document.segment_count,
            "text_length": len(self.document.text),
            "outputs": [output.to_dict() for output in self.outputs],
        }


@dataclass(slots=True, frozen=True)
class TranscriptionBatchResult:
    artifacts: tuple[TranscriptionArtifact, ...]

    @property
    def generated_files(self) -> tuple[Path, ...]:
        files: list[Path] = []
        for artifact in self.artifacts:
            files.extend(artifact.generated_files)
        return tuple(files)

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": len(self.artifacts),
            "generated_file_count": len(self.generated_files),
            "generated_files": [str(path) for path in self.generated_files],
            "artifacts": [artifact.to_dict() for artifact in self.artifacts],
        }
