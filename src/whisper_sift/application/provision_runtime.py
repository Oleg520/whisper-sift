from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class ProvisionTranscriptionRuntimeRequest:
    pass


@dataclass(slots=True)
class ProvisionTranscriptionRuntimeResult:
    ready: bool = True


def run_provision_transcription_runtime(
    request: ProvisionTranscriptionRuntimeRequest | None = None,
) -> ProvisionTranscriptionRuntimeResult:
    from whisper_sift.runtime.dependencies import ensure_transcription_dependencies

    ensure_transcription_dependencies()
    return ProvisionTranscriptionRuntimeResult(ready=True)
