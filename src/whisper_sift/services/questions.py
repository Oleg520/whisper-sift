from __future__ import annotations

from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.domain.extraction import extract_questions
from whisper_sift.infrastructure.filesystem import (
    build_question_output_path,
    ensure_existing_file,
    read_text_file,
    write_text_file,
)
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


def extract_questions_from_files(
    options: QuestionExtractionOptions,
    *,
    reporter: ProgressReporter | None = None,
) -> list[Path]:
    generated_files: list[Path] = []

    for source in options.files:
        resolved_source = ensure_existing_file(
            source,
            error_prefix="Transcript file not found",
        )

        extraction = extract_questions(
            read_text_file(resolved_source),
            deduplicate=options.deduplicate,
            min_length=options.min_length,
            max_length=options.max_length,
            interviewer_labels=options.interviewer_labels,
            source_name=resolved_source.name,
        )
        output_path = build_question_output_path(
            source=resolved_source,
            output_dir=options.output_dir,
            suffix=options.suffix,
        )
        write_text_file(output_path, "\n".join(extraction.question_texts))

        report_progress(
            reporter,
            f"[questions] {resolved_source.name} -> {output_path.name} "
            f"({len(extraction.questions)} items)",
        )
        generated_files.append(output_path)

    return generated_files
