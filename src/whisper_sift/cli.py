from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from whisper_sift.config import QuestionExtractionOptions, TranscriptionOptions


COMMANDS = {"transcribe", "extract-questions", "pipeline"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="whisper-sift",
        description=(
            "CLI для расшифровки интервью через Whisper и извлечения вопросов "
            "интервьюеров из transcript-файлов."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    transcribe_parser = subparsers.add_parser(
        "transcribe",
        help="Расшифровать медиафайлы",
    )
    _add_transcription_arguments(transcribe_parser)
    transcribe_parser.set_defaults(handler=_handle_transcribe)

    questions_parser = subparsers.add_parser(
        "extract-questions",
        help="Извлечь вопросы из .txt-расшифровок",
    )
    _add_question_arguments(questions_parser)
    questions_parser.set_defaults(handler=_handle_extract_questions)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Расшифровать файлы и сразу извлечь вопросы",
    )
    _add_transcription_arguments(pipeline_parser)
    pipeline_parser.add_argument(
        "--questions-dir",
        type=Path,
        default=None,
        help="Папка для файлов с вопросами; по умолчанию используется --output-dir.",
    )
    pipeline_parser.add_argument(
        "--suffix",
        default="_questions.txt",
        help="Суффикс имени файлов с вопросами.",
    )
    pipeline_parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Не удалять дубликаты вопросов.",
    )
    pipeline_parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Минимальная длина вопроса.",
    )
    pipeline_parser.add_argument(
        "--max-length",
        type=int,
        default=240,
        help="Максимальная длина вопроса.",
    )
    pipeline_parser.set_defaults(handler=_handle_pipeline)

    return parser


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="+", type=Path, help="Медиафайлы для расшифровки.")
    parser.add_argument(
        "--model",
        default="small",
        help="Название модели Whisper, например tiny/base/small/medium/large.",
    )
    parser.add_argument(
        "--language",
        default="ru",
        help="Код языка. Укажите auto для автоопределения.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Устройство для запуска модели: auto/cpu/cuda/mps. По умолчанию auto.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Папка для результатов расшифровки.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["txt", "srt"],
        help="Выходные форматы. По умолчанию: txt srt.",
    )


def _add_question_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="+", type=Path, help="TXT-файлы с расшифровками.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка для файлов с вопросами. По умолчанию рядом с источником.",
    )
    parser.add_argument(
        "--suffix",
        default="_questions.txt",
        help="Суффикс выходного файла.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Не удалять дубликаты вопросов.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Минимальная длина вопроса.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=240,
        help="Максимальная длина вопроса.",
    )


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    if not normalized:
        return normalized

    first_arg = normalized[0]
    if first_arg in COMMANDS or first_arg.startswith("-"):
        return normalized

    return ["transcribe", *normalized]


def _handle_transcribe(args: argparse.Namespace) -> int:
    from whisper_sift.runtime.dependencies import ensure_transcription_dependencies

    ensure_transcription_dependencies()
    from whisper_sift.services.transcription import transcribe_files

    options = TranscriptionOptions(
        files=args.files,
        output_dir=args.output_dir.resolve(),
        model=args.model,
        language=None if args.language.lower() == "auto" else args.language,
        device=args.device,
        formats=tuple(args.formats),
    )
    transcribe_files(options)
    return 0


def _handle_extract_questions(args: argparse.Namespace) -> int:
    from whisper_sift.services.questions import extract_questions_from_files

    options = QuestionExtractionOptions(
        files=args.files,
        output_dir=args.output_dir.resolve() if args.output_dir else None,
        suffix=args.suffix,
        deduplicate=not args.no_deduplicate,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    extract_questions_from_files(options)
    return 0


def _handle_pipeline(args: argparse.Namespace) -> int:
    from whisper_sift.runtime.dependencies import ensure_transcription_dependencies

    ensure_transcription_dependencies()
    from whisper_sift.services.questions import extract_questions_from_files
    from whisper_sift.services.transcription import transcribe_files

    transcript_dir = args.output_dir.resolve()
    question_dir = args.questions_dir.resolve() if args.questions_dir else transcript_dir

    transcription_options = TranscriptionOptions(
        files=args.files,
        output_dir=transcript_dir,
        model=args.model,
        language=None if args.language.lower() == "auto" else args.language,
        device=args.device,
        formats=tuple(args.formats),
    )
    generated_files = transcribe_files(transcription_options)

    transcript_files = [path for path in generated_files if path.suffix.lower() == ".txt"]
    question_options = QuestionExtractionOptions(
        files=transcript_files,
        output_dir=question_dir,
        suffix=args.suffix,
        deduplicate=not args.no_deduplicate,
        min_length=args.min_length,
        max_length=args.max_length,
    )
    extract_questions_from_files(question_options)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = _normalize_argv(sys.argv[1:] if argv is None else argv)
    if not normalized_argv:
        parser.print_help()
        return 1

    args = parser.parse_args(normalized_argv)
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    return handler(args)
