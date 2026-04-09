from __future__ import annotations

from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.utils.text import extract_question_candidates


def extract_questions_from_files(options: QuestionExtractionOptions) -> list[Path]:
    generated_files: list[Path] = []

    for source in options.files:
        resolved_source = source.resolve()
        if not resolved_source.exists():
            raise FileNotFoundError(f"Transcript file not found: {resolved_source}")

        questions = extract_question_candidates(
            resolved_source.read_text(encoding="utf-8"),
            deduplicate=options.deduplicate,
            min_length=options.min_length,
            max_length=options.max_length,
            interviewer_labels=options.interviewer_labels,
        )
        output_path = _build_output_path(
            source=resolved_source,
            output_dir=options.output_dir,
            suffix=options.suffix,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(questions), encoding="utf-8")

        print(
            f"[questions] {resolved_source.name} -> {output_path.name} "
            f"({len(questions)} items)"
        )
        generated_files.append(output_path)

    return generated_files


def _build_output_path(source: Path, output_dir: Path | None, suffix: str) -> Path:
    target_dir = output_dir.resolve() if output_dir else source.parent
    return target_dir / f"{source.stem}{suffix}"
