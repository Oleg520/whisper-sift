from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.domain.extraction import extract_questions
from whisper_sift.infrastructure.filesystem import (
    build_json_sidecar_path,
    build_question_output_path,
    ensure_existing_file,
    read_text_file,
    write_json_file,
    write_text_file,
)
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


@dataclass(slots=True)
class QuestionOutputArtifacts:
    text_files: tuple[Path, ...]
    json_files: tuple[Path, ...]


def extract_questions_from_files(
    options: QuestionExtractionOptions,
    *,
    reporter: ProgressReporter | None = None,
) -> QuestionOutputArtifacts:
    generated_text_files: list[Path] = []
    generated_json_files: list[Path] = []

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
        generated_text_files.append(output_path)

        json_output_path: Path | None = None
        if options.write_json:
            json_output_path = build_json_sidecar_path(output_path)
            write_json_file(
                json_output_path,
                extraction.to_dict(source_path=str(resolved_source)),
            )
            generated_json_files.append(json_output_path)

        details = f"[questions] {resolved_source.name} -> {output_path.name}"
        if json_output_path is not None:
            details += f" + {json_output_path.name}"
        report_progress(
            reporter,
            f"{details} ({len(extraction.questions)} items)",
        )

    return QuestionOutputArtifacts(
        text_files=tuple(generated_text_files),
        json_files=tuple(generated_json_files),
    )
