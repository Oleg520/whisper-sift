from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import whisper
from whisper.utils import get_writer

from whisper_sift.config import TranscriptionOptions
from whisper_sift.infrastructure.ffmpeg import ensure_ffmpeg_on_path


def transcribe_files(options: TranscriptionOptions) -> list[Path]:
    output_dir = options.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = ensure_ffmpeg_on_path()
    resolved_device = _resolve_device(options.device)
    use_fp16 = resolved_device == "cuda"

    print(f"[ffmpeg]  {ffmpeg_exe}")
    print(f"[model]   {options.model}")
    print(f"[device]  requested={options.device} resolved={resolved_device}")
    print(f"[fp16]    {use_fp16}")

    model = whisper.load_model(options.model, device=resolved_device)
    generated_files: list[Path] = []

    for source in options.files:
        resolved_source = source.resolve()
        if not resolved_source.exists():
            raise FileNotFoundError(f"Input file not found: {resolved_source}")

        print(f"[start] {resolved_source.name}")
        result = model.transcribe(
            str(resolved_source),
            task="transcribe",
            language=options.language,
            fp16=use_fp16,
            verbose=False,
            temperature=0,
        )
        generated_files.extend(
            _write_outputs(
                result=result,
                source=resolved_source,
                output_dir=output_dir,
                formats=options.formats,
            )
        )
        print(f"[done]  {resolved_source.name}")

    return generated_files


def _resolve_device(requested_device: str) -> str:
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
        print("[device]  CUDA requested, but unavailable. Falling back to cpu.")
        return "cpu"

    if normalized == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        print("[device]  MPS requested, but unavailable. Falling back to cpu.")
        return "cpu"

    if normalized == "cpu":
        return "cpu"

    print(f"[device]  Unknown device '{requested_device}'. Falling back to cpu.")
    return "cpu"


def _write_outputs(
    result: dict[str, Any],
    source: Path,
    output_dir: Path,
    formats: tuple[str, ...],
) -> list[Path]:
    writer_options = {
        "highlight_words": False,
        "max_line_count": None,
        "max_line_width": None,
    }
    generated_files: list[Path] = []

    for output_format in formats:
        writer = get_writer(output_format, str(output_dir))
        writer(result, str(source), writer_options)
        generated_files.append(output_dir / f"{source.stem}.{output_format}")

    return generated_files
