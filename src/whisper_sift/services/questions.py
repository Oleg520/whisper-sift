from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.domain.extraction import extract_questions
from whisper_sift.domain.questions import QuestionOutputArtifact
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
    items: tuple[QuestionOutputArtifact, ...]

    @property
    def text_files(self) -> tuple[Path, ...]:
        return tuple(item.text_file for item in self.items)

    @property
    def json_files(self) -> tuple[Path, ...]:
        return tuple(
            item.json_file
            for item in self.items
            if item.json_file is not None
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "file_count": len(self.items),
            "text_files": [str(path) for path in self.text_files],
            "json_files": [str(path) for path in self.json_files],
            "items": [item.to_dict() for item in self.items],
        }


def extract_questions_from_files(
    options: QuestionExtractionOptions,
    *,
    reporter: ProgressReporter | None = None,
) -> QuestionOutputArtifacts:
    generated_items: list[QuestionOutputArtifact] = []

    for source in options.files:
        resolved_source = ensure_existing_file(
            source,
            error_prefix="Transcript file not found",
        )

        extraction = extract_questions(
            read_text_file(resolved_source),
            deduplicate=options.extraction.deduplicate,
            min_length=options.extraction.min_length,
            max_length=options.extraction.max_length,
            interviewer_labels=options.extraction.interviewer_labels,
            source_name=resolved_source.name,
        )
        output_path = build_question_output_path(
            source=resolved_source,
            output_dir=options.output_dir,
            suffix=options.output.suffix,
        )
        write_text_file(output_path, "\n".join(extraction.question_texts))

        json_output_path: Path | None = None
        if options.output.write_json:
            json_output_path = build_json_sidecar_path(output_path)
            write_json_file(
                json_output_path,
                extraction.to_dict(source_path=str(resolved_source)),
            )

        details = f"[questions] {resolved_source.name} -> {output_path.name}"
        if json_output_path is not None:
            details += f" + {json_output_path.name}"
        report_progress(
            reporter,
            f"{details} ({len(extraction.questions)} items)",
        )
        generated_items.append(
            QuestionOutputArtifact(
                source_path=resolved_source,
                text_file=output_path,
                json_file=json_output_path,
                question_count=len(extraction.questions),
            )
        )

    return QuestionOutputArtifacts(items=tuple(generated_items))
