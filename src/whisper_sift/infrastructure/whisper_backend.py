from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from whisper_sift.config import FAKE_TRANSCRIPTION_FILE_ENV, FAKE_TRANSCRIPTION_TEXT_ENV
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


class WhisperBackend(Protocol):
    model_name: str
    resolved_device: str
    use_fp16: bool
    requires_media_runtime: bool

    def transcribe_file(self, source: Path, *, language: str | None) -> dict[str, Any]:
        ...


@dataclass(slots=True)
class OpenAIWhisperBackend:
    model_name: str
    resolved_device: str
    use_fp16: bool
    model: Any
    requires_media_runtime: bool = True

    def transcribe_file(self, source: Path, *, language: str | None) -> dict[str, Any]:
        return self.model.transcribe(
            str(source),
            task="transcribe",
            language=language,
            fp16=self.use_fp16,
            verbose=False,
            temperature=0,
        )


@dataclass(slots=True, frozen=True)
class FixtureWhisperBackend:
    model_name: str
    resolved_device: str
    use_fp16: bool
    transcript_text: str
    requires_media_runtime: bool = False

    def transcribe_file(self, source: Path, *, language: str | None) -> dict[str, Any]:
        return _build_fixture_result(self.transcript_text, language=language)


def load_whisper_backend(
    model_name: str,
    requested_device: str,
    *,
    reporter: ProgressReporter | None = None,
) -> WhisperBackend:
    fixture_text = load_fake_transcription_text()
    if fixture_text is not None:
        report_progress(
            reporter,
            "[backend] fixture transcription backend enabled",
        )
        return FixtureWhisperBackend(
            model_name=model_name,
            resolved_device="cpu",
            use_fp16=False,
            transcript_text=fixture_text,
        )

    import whisper

    resolved_device = resolve_backend_device(requested_device, reporter=reporter)
    use_fp16 = resolved_device == "cuda"
    model = whisper.load_model(model_name, device=resolved_device)
    return OpenAIWhisperBackend(
        model_name=model_name,
        resolved_device=resolved_device,
        use_fp16=use_fp16,
        model=model,
    )


def resolve_backend_device(
    requested_device: str,
    *,
    reporter: ProgressReporter | None = None,
    torch_module: Any | None = None,
) -> str:
    normalized = requested_device.strip().lower()
    torch = torch_module

    if torch is None:
        import torch as torch_module_imported

        torch = torch_module_imported

    if normalized == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    if normalized == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        report_progress(reporter, "[device]  CUDA requested, but unavailable. Falling back to cpu.")
        return "cpu"

    if normalized == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        report_progress(reporter, "[device]  MPS requested, but unavailable. Falling back to cpu.")
        return "cpu"

    if normalized == "cpu":
        return "cpu"

    report_progress(reporter, f"[device]  Unknown device '{requested_device}'. Falling back to cpu.")
    return "cpu"


def load_fake_transcription_text() -> str | None:
    fixture_path = os.environ.get(FAKE_TRANSCRIPTION_FILE_ENV)
    if fixture_path:
        path = Path(fixture_path).expanduser()
        if path.exists():
            return path.read_text(encoding="utf-8")

    fixture_text = os.environ.get(FAKE_TRANSCRIPTION_TEXT_ENV)
    if fixture_text:
        return fixture_text

    return None


def is_fake_transcription_enabled() -> bool:
    return load_fake_transcription_text() is not None


def _build_fixture_result(text: str, *, language: str | None) -> dict[str, Any]:
    normalized_text = text.strip()
    segments: list[dict[str, Any]] = []

    for index, raw_line in enumerate(line for line in normalized_text.splitlines() if line.strip()):
        start = float(index)
        end = float(index + 1)
        segments.append(
            {
                "id": index,
                "seek": 0,
                "start": start,
                "end": end,
                "text": f" {raw_line.strip()}",
                "tokens": [],
                "temperature": 0.0,
                "avg_logprob": 0.0,
                "compression_ratio": 0.0,
                "no_speech_prob": 0.0,
            }
        )

    return {
        "text": normalized_text,
        "segments": segments,
        "language": language or "und",
    }
