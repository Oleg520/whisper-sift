from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import TranscriptionOptions
from whisper_sift.runtime.reporting import ProgressReporter


@dataclass(slots=True)
class TranscribeRequest:
    options: TranscriptionOptions
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class TranscribeResult:
    generated_files: tuple[Path, ...]


def run_transcribe(request: TranscribeRequest) -> TranscribeResult:
    from whisper_sift.runtime.dependencies import ensure_transcription_dependencies
    from whisper_sift.services.transcription import transcribe_files

    ensure_transcription_dependencies()
    generated_files = transcribe_files(request.options, reporter=request.reporter)
    return TranscribeResult(generated_files=tuple(generated_files))
