from __future__ import annotations

import argparse

from whisper_sift.application.pipeline import PipelineRequest, run_pipeline
from whisper_sift.application.provision_runtime import (
    ProvisionTranscriptionRuntimeRequest,
    run_provision_transcription_runtime,
)
from whisper_sift.cli_handlers.shared import normalize_pipeline_formats
from whisper_sift.config import TranscriptionOptions, normalize_transcription_language
from whisper_sift.runtime.reporting import ConsoleReporter


def handle_pipeline(args: argparse.Namespace) -> int:
    reporter = ConsoleReporter()
    run_provision_transcription_runtime(ProvisionTranscriptionRuntimeRequest())
    transcript_dir = args.output_dir.resolve()
    question_dir = args.questions_dir.resolve() if args.questions_dir else transcript_dir
    normalized_formats, txt_added = normalize_pipeline_formats(args.formats)

    if txt_added:
        print(
            "[pipeline] Questions require txt transcripts. "
            "Added 'txt' to the requested output formats."
        )

    transcription_options = TranscriptionOptions(
        files=args.files,
        output_dir=transcript_dir,
        model=args.model,
        language=normalize_transcription_language(args.language),
        device=args.device,
        formats=normalized_formats,
    )
    run_pipeline(
        PipelineRequest(
            transcription_options=transcription_options,
            questions_output_dir=question_dir,
            suffix=args.suffix,
            write_json=args.json,
            deduplicate=not args.no_deduplicate,
            min_length=args.min_length,
            max_length=args.max_length,
            interviewer_labels=tuple(args.interviewer_label),
            reporter=reporter,
        )
    )
    return 0
