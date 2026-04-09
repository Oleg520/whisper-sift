from __future__ import annotations

from pathlib import Path
from typing import Any


def ensure_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def ensure_existing_file(path: Path, *, error_prefix: str) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{error_prefix}: {resolved}")
    return resolved


def read_text_file(path: Path, *, encoding: str = "utf-8") -> str:
    return path.read_text(encoding=encoding)


def write_text_file(path: Path, content: str, *, encoding: str = "utf-8") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding=encoding)
    return path


def build_question_output_path(source: Path, output_dir: Path | None, suffix: str) -> Path:
    target_dir = output_dir.resolve() if output_dir else source.parent
    return target_dir / f"{source.stem}{suffix}"


def write_whisper_outputs(
    *,
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
        if output_format == "txt":
            _write_txt_output(
                result=result,
                source=source,
                output_dir=output_dir,
            )
        else:
            from whisper.utils import get_writer

            writer = get_writer(output_format, str(output_dir))
            writer(result, str(source), writer_options)
        generated_files.append(output_dir / f"{source.stem}.{output_format}")

    return generated_files


def _write_txt_output(*, result: dict[str, Any], source: Path, output_dir: Path) -> None:
    text = str(result.get("text", "")).strip()
    target_path = output_dir / f"{source.stem}.txt"
    target_path.write_text(f"{text}\n" if text else "", encoding="utf-8")
