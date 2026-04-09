from __future__ import annotations

from pathlib import Path

from whisper_sift.config import TranscriptionOptions
from whisper_sift.infrastructure.ffmpeg import ensure_ffmpeg_on_path
from whisper_sift.infrastructure.filesystem import (
    ensure_existing_file,
    ensure_output_dir,
    write_whisper_outputs,
)
from whisper_sift.infrastructure.whisper_backend import load_whisper_backend
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

    backend = load_whisper_backend(
        options.model,
        options.device,
        reporter=reporter,
    )

    if backend.requires_media_runtime:
        ffmpeg_exe = ensure_ffmpeg_on_path()
        report_progress(reporter, f"[ffmpeg]  {ffmpeg_exe}")
    else:
        report_progress(reporter, "[ffmpeg]  skipped (fixture backend)")

    report_progress(reporter, f"[model]   {backend.model_name}")
    report_progress(
        reporter,
        f"[device]  requested={options.device} resolved={backend.resolved_device}",
    )
    report_progress(reporter, f"[fp16]    {backend.use_fp16}")
    generated_files: list[Path] = []

    for resolved_source in resolved_sources:
        report_progress(reporter, f"[start] {resolved_source.name}")
        result = backend.transcribe_file(
            resolved_source,
            language=options.language,
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
