from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from whisper_sift.config import (
    AUTO_DETECT_LANGUAGE,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_OUTPUT_FORMATS,
    DEFAULT_QUESTION_SUFFIX,
    DEFAULT_TRANSCRIPTION_DEVICE,
    DEFAULT_TRANSCRIPTION_LANGUAGE,
    DEFAULT_TRANSCRIPTION_MODEL,
    DEFAULT_WRITE_QUESTION_JSON,
    QuestionExtractionOptions,
    TranscriptionOptions,
    normalize_transcription_language,
)
from whisper_sift.runtime.reporting import ConsoleReporter


EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_FILE_NOT_FOUND = 3
EXIT_DEPENDENCY_ERROR = 4
EXIT_INTERRUPTED = 130

COMMANDS = {"transcribe", "extract-questions", "pipeline", "doctor"}
DEFAULT_OUTPUT_FORMATS_TEXT = " ".join(DEFAULT_OUTPUT_FORMATS)


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
        help="Извлечь вопросы из transcript-файлов (.txt, .srt)",
    )
    _add_question_arguments(questions_parser)
    questions_parser.set_defaults(handler=_handle_extract_questions)

    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Проверить окружение и готовность транскрибации",
    )
    doctor_parser.add_argument(
        "--install-missing",
        action="store_true",
        help="Попробовать установить отсутствующие runtime-зависимости перед повторной проверкой.",
    )
    doctor_parser.set_defaults(handler=_handle_doctor)

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
        default=DEFAULT_QUESTION_SUFFIX,
        help="Суффикс имени файлов с вопросами.",
    )
    pipeline_parser.add_argument(
        "--json",
        action="store_true",
        default=DEFAULT_WRITE_QUESTION_JSON,
        help="Сохранить дополнительный JSON-файл со структурированными данными по вопросам.",
    )
    pipeline_parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Не удалять дубликаты вопросов.",
    )
    pipeline_parser.add_argument(
        "--interviewer-label",
        action="append",
        default=[],
        help=(
            "Явная speaker label интервьюера для speaker-labeled transcript-файлов. "
            "Можно указать несколько раз."
        ),
    )
    pipeline_parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_QUESTION_LENGTH,
        help="Минимальная длина вопроса.",
    )
    pipeline_parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_QUESTION_LENGTH,
        help="Максимальная длина вопроса.",
    )
    pipeline_parser.set_defaults(handler=_handle_pipeline)

    return parser


def _add_transcription_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("files", nargs="+", type=Path, help="Медиафайлы для расшифровки.")
    parser.add_argument(
        "--model",
        default=DEFAULT_TRANSCRIPTION_MODEL,
        help="Название модели Whisper, например tiny/base/small/medium/large.",
    )
    parser.add_argument(
        "--language",
        default=DEFAULT_TRANSCRIPTION_LANGUAGE,
        help=f"Код языка. Укажите {AUTO_DETECT_LANGUAGE} для автоопределения.",
    )
    parser.add_argument(
        "--device",
        default=DEFAULT_TRANSCRIPTION_DEVICE,
        help=(
            "Устройство для запуска модели: auto/cpu/cuda/mps. "
            f"По умолчанию {DEFAULT_TRANSCRIPTION_DEVICE}."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Папка для результатов расшифровки.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=list(DEFAULT_OUTPUT_FORMATS),
        help=f"Выходные форматы. По умолчанию: {DEFAULT_OUTPUT_FORMATS_TEXT}.",
    )


def _add_question_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Transcript-файлы с расшифровками (.txt, .srt).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Папка для файлов с вопросами. По умолчанию рядом с источником.",
    )
    parser.add_argument(
        "--suffix",
        default=DEFAULT_QUESTION_SUFFIX,
        help="Суффикс выходного файла.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=DEFAULT_WRITE_QUESTION_JSON,
        help="Сохранить дополнительный JSON-файл со структурированными данными по вопросам.",
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Не удалять дубликаты вопросов.",
    )
    parser.add_argument(
        "--interviewer-label",
        action="append",
        default=[],
        help=(
            "Явная speaker label интервьюера, например 'Интервьюер', "
            "'Interviewer' или 'SPEAKER_00'. Можно указать несколько раз."
        ),
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=DEFAULT_MIN_QUESTION_LENGTH,
        help="Минимальная длина вопроса.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=DEFAULT_MAX_QUESTION_LENGTH,
        help="Максимальная длина вопроса.",
    )


def _normalize_argv(argv: Sequence[str]) -> list[str]:
    normalized = list(argv)
    if not normalized:
        return normalized

    first_arg = normalized[0]
    if first_arg in COMMANDS or first_arg.startswith("-"):
        return normalized

    if _looks_like_transcription_target(first_arg):
        return ["transcribe", *normalized]

    return normalized


def _looks_like_transcription_target(value: str) -> bool:
    candidate = Path(value)
    if candidate.exists():
        return True
    if candidate.suffix:
        return True
    return "\\" in value or "/" in value


def _handle_transcribe(args: argparse.Namespace) -> int:
    from whisper_sift.application.transcribe import TranscribeRequest, run_transcribe

    reporter = ConsoleReporter()
    normalized_formats = _normalize_output_formats(args.formats)
    options = TranscriptionOptions(
        files=args.files,
        output_dir=args.output_dir.resolve(),
        model=args.model,
        language=normalize_transcription_language(args.language),
        device=args.device,
        formats=normalized_formats,
    )
    run_transcribe(TranscribeRequest(options=options, reporter=reporter))
    return EXIT_SUCCESS


def _handle_extract_questions(args: argparse.Namespace) -> int:
    from whisper_sift.application.extract_questions import (
        ExtractQuestionsRequest,
        run_extract_questions,
    )

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
    return EXIT_SUCCESS


def _handle_pipeline(args: argparse.Namespace) -> int:
    from whisper_sift.application.pipeline import PipelineRequest, run_pipeline

    reporter = ConsoleReporter()
    transcript_dir = args.output_dir.resolve()
    question_dir = args.questions_dir.resolve() if args.questions_dir else transcript_dir
    normalized_formats, txt_added = _normalize_pipeline_formats(args.formats)

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
    return EXIT_SUCCESS


def _handle_doctor(args: argparse.Namespace) -> int:
    from whisper_sift.application.doctor import DoctorRequest, run_doctor
    from whisper_sift.runtime.doctor import print_doctor_report

    report = run_doctor(DoctorRequest(install_missing=args.install_missing)).report
    print_doctor_report(report)
    return EXIT_SUCCESS if report.is_ready else EXIT_RUNTIME_ERROR


def _normalize_output_formats(formats: Sequence[str]) -> tuple[str, ...]:
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


def _normalize_pipeline_formats(formats: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    normalized_formats = list(_normalize_output_formats(formats))
    if "txt" in normalized_formats:
        return tuple(normalized_formats), False

    normalized_formats.append("txt")
    return tuple(normalized_formats), True


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    normalized_argv = _normalize_argv(sys.argv[1:] if argv is None else argv)
    if not normalized_argv:
        parser.print_help()
        return EXIT_USAGE_ERROR

    try:
        args = parser.parse_args(normalized_argv)
    except SystemExit as exc:
        return int(exc.code)

    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return EXIT_USAGE_ERROR

    try:
        return handler(args)
    except KeyboardInterrupt:
        print("[error] Operation cancelled by user.", file=sys.stderr)
        return EXIT_INTERRUPTED
    except FileNotFoundError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_FILE_NOT_FOUND
    except ModuleNotFoundError as exc:
        print(f"[error] Missing Python module: {exc.name}", file=sys.stderr)
        return EXIT_DEPENDENCY_ERROR
    except subprocess.CalledProcessError as exc:
        command = _format_command(exc.cmd)
        print(
            f"[error] External command failed with exit code {exc.returncode}: {command}",
            file=sys.stderr,
        )
        return EXIT_DEPENDENCY_ERROR
    except RuntimeError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR
    except Exception as exc:
        print(f"[error] Unexpected failure: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


def _format_command(command: Sequence[str] | str | None) -> str:
    if command is None:
        return "<unknown>"
    if isinstance(command, str):
        return command
    return " ".join(str(part) for part in command)
