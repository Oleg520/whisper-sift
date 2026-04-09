from __future__ import annotations

from pathlib import Path

import torch
import whisper

from whisper_sift.config import TranscriptionOptions
from whisper_sift.infrastructure.ffmpeg import ensure_ffmpeg_on_path
from whisper_sift.infrastructure.filesystem import ensure_existing_file, ensure_output_dir, write_whisper_outputs
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


def transcribe_files(
    options: TranscriptionOptions,
    *,
    reporter: ProgressReporter | None = None,
) -> list[Path]:
    resolved_sources = [
        ensure_existing_file(source, error_prefix="Input file not found")
        for source in options.files
    ]
    output_dir = ensure_output_dir(options.output_dir)

    ffmpeg_exe = ensure_ffmpeg_on_path()
    resolved_device = _resolve_device(options.device, reporter=reporter)
    use_fp16 = resolved_device == "cuda"

    report_progress(reporter, f"[ffmpeg]  {ffmpeg_exe}")
    report_progress(reporter, f"[model]   {options.model}")
    report_progress(reporter, f"[device]  requested={options.device} resolved={resolved_device}")
    report_progress(reporter, f"[fp16]    {use_fp16}")

    model = whisper.load_model(options.model, device=resolved_device)
    generated_files: list[Path] = []

    for resolved_source in resolved_sources:
        report_progress(reporter, f"[start] {resolved_source.name}")
        result = model.transcribe(
            str(resolved_source),
            task="transcribe",
            language=options.language,
            fp16=use_fp16,
            verbose=False,
            temperature=0,
        )
        generated_files.extend(
            write_whisper_outputs(
                result=result,
                source=resolved_source,
                output_dir=output_dir,
                formats=options.formats,
            )
        )
        report_progress(reporter, f"[done]  {resolved_source.name}")

    return generated_files


def _resolve_device(
    requested_device: str,
    *,
    reporter: ProgressReporter | None = None,
) -> str:
    normalized = requested_device.strip().lower()

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
