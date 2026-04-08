from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


DEFAULT_OUTPUT_FORMATS = ("txt", "srt")


@dataclass(slots=True)
class TranscriptionOptions:
    files: list[Path]
    output_dir: Path
    model: str = "small"
    language: str | None = "ru"
    device: str = "auto"
    formats: tuple[str, ...] = DEFAULT_OUTPUT_FORMATS


@dataclass(slots=True)
class QuestionExtractionOptions:
    files: list[Path]
    output_dir: Path | None = None
    suffix: str = "_questions.txt"
    deduplicate: bool = True
    min_length: int = 10
    max_length: int = 240
