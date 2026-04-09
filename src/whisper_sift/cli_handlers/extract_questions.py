from __future__ import annotations

import argparse

from whisper_sift.application.extract_questions import (
    ExtractQuestionsRequest,
    run_extract_questions,
)
from whisper_sift.config import QuestionExtractionOptions
from whisper_sift.runtime.reporting import ConsoleReporter


def handle_extract_questions(args: argparse.Namespace) -> int:
    reporter = ConsoleReporter()
    options = QuestionExtractionOptions(
        files=args.files,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        suffix=args.suffix,
        write_json=args.json,
        deduplicate=not args.no_deduplicate,
        min_length=args.min_length,
        max_length=args.max_length,
        interviewer_labels=tuple(args.interviewer_label),
    )
    run_extract_questions(
        ExtractQuestionsRequest(options=options, reporter=reporter)
    )
    return 0
