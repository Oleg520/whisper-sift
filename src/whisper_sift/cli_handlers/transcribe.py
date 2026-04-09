from __future__ import annotations

import argparse

from whisper_sift.application.provision_runtime import (
    ProvisionTranscriptionRuntimeRequest,
    run_provision_transcription_runtime,
)
from whisper_sift.application.transcribe import TranscribeRequest, run_transcribe
from whisper_sift.cli_handlers.shared import normalize_output_formats
from whisper_sift.config import TranscriptionOptions, normalize_transcription_language
from whisper_sift.runtime.reporting import ConsoleReporter


def handle_transcribe(args: argparse.Namespace) -> int:
    reporter = ConsoleReporter()
    run_provision_transcription_runtime(ProvisionTranscriptionRuntimeRequest())
    normalized_formats = normalize_output_formats(args.formats)
    options = TranscriptionOptions(
        files=args.files,
        output_dir=args.output_dir.resolve(),
        model=args.model,
        language=normalize_transcription_language(args.language),
        device=args.device,
        formats=normalized_formats,
    )
    run_transcribe(
        TranscribeRequest(
            options=options,
            summary_json_path=args.summary_json,
            reporter=reporter,
        )
    )
    return 0
