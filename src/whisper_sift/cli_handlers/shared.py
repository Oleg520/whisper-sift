from __future__ import annotations

from pathlib import Path
from typing import Sequence


def normalize_output_formats(formats: Sequence[str]) -> tuple[str, ...]:
    normalized_formats: list[str] = []
    seen: set[str] = set()

    for output_format in formats:
        normalized = output_format.strip().lower()
        if not normalized or normalized in seen:
            continue
        normalized_formats.append(normalized)
        seen.add(normalized)

    if not normalized_formats:
        raise RuntimeError("At least one output format is required.")

    return tuple(normalized_formats)


def normalize_pipeline_formats(formats: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    normalized_formats = list(normalize_output_formats(formats))
    if "txt" in normalized_formats:
        return tuple(normalized_formats), False

    normalized_formats.append("txt")
    return tuple(normalized_formats), True


def default_golden_set_path() -> Path:
    return Path.cwd() / ".analysis" / "eval" / "golden_set.json"
