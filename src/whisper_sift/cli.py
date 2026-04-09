from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from whisper_sift.cli_handlers import (
    handle_doctor as _command_handle_doctor,
    handle_evaluate as _command_handle_evaluate,
    handle_extract_questions as _command_handle_extract_questions,
    handle_pipeline as _command_handle_pipeline,
    handle_transcribe as _command_handle_transcribe,
)
from whisper_sift.cli_handlers.shared import (
    default_golden_set_path as _shared_default_golden_set_path,
    normalize_output_formats as _shared_normalize_output_formats,
    normalize_pipeline_formats as _shared_normalize_pipeline_formats,
)
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
)

EXIT_SUCCESS = 0
EXIT_RUNTIME_ERROR = 1
EXIT_USAGE_ERROR = 2
EXIT_FILE_NOT_FOUND = 3
EXIT_DEPENDENCY_ERROR = 4
EXIT_INTERRUPTED = 130

COMMANDS = {"transcribe", "extract-questions", "pipeline", "doctor", "evaluate"}
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
    transcribe_parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON summary по результатам транскрибации.",
    )
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

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Проверить качество extraction по локальному golden set",
    )
    evaluate_parser.add_argument(
        "--golden-set",
        type=Path,
        default=_default_golden_set_path(),
        help="Путь до JSON golden set для локальной оценки качества extraction.",
    )
    evaluate_parser.add_argument(
        "--report-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON-отчёт; по умолчанию рядом с golden set.",
    )
    evaluate_parser.add_argument(
        "--baseline-report",
        type=Path,
        default=None,
        help=(
            "Путь до baseline JSON-отчёта. Если указан, evaluate сравнит текущий "
            "результат с baseline."
        ),
    )
    evaluate_parser.add_argument(
        "--diff-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON-дифф между текущим отчётом и baseline.",
    )
    evaluate_parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="После оценки обновить baseline текущим результатом.",
    )
    evaluate_parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="Имя конкретного evaluation-кейса. Можно указать несколько раз.",
    )
    evaluate_parser.set_defaults(handler=_handle_evaluate)

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
        "--summary-json",
        type=Path,
        default=None,
        help="Куда сохранить JSON summary по результатам полного пайплайна.",
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
    return _cli_handle_transcribe(args)


def _handle_extract_questions(args: argparse.Namespace) -> int:
    return _cli_handle_extract_questions(args)


def _handle_pipeline(args: argparse.Namespace) -> int:
    return _cli_handle_pipeline(args)


def _handle_doctor(args: argparse.Namespace) -> int:
    return _cli_handle_doctor(args)


def _handle_evaluate(args: argparse.Namespace) -> int:
    return _cli_handle_evaluate(args)


def _normalize_output_formats(formats: Sequence[str]) -> tuple[str, ...]:
    return _shared_normalize_output_formats(formats)


def _normalize_pipeline_formats(formats: Sequence[str]) -> tuple[tuple[str, ...], bool]:
    return _shared_normalize_pipeline_formats(formats)


def _default_golden_set_path() -> Path:
    return _shared_default_golden_set_path()


def _cli_handle_transcribe(args: argparse.Namespace) -> int:
    return EXIT_SUCCESS if _command_handle_transcribe(args) == 0 else EXIT_RUNTIME_ERROR


def _cli_handle_extract_questions(args: argparse.Namespace) -> int:
    return EXIT_SUCCESS if _command_handle_extract_questions(args) == 0 else EXIT_RUNTIME_ERROR


def _cli_handle_pipeline(args: argparse.Namespace) -> int:
    return EXIT_SUCCESS if _command_handle_pipeline(args) == 0 else EXIT_RUNTIME_ERROR


def _cli_handle_doctor(args: argparse.Namespace) -> int:
    return EXIT_SUCCESS if _command_handle_doctor(args) == 0 else EXIT_RUNTIME_ERROR


def _cli_handle_evaluate(args: argparse.Namespace) -> int:
    return EXIT_SUCCESS if _command_handle_evaluate(args) == 0 else EXIT_RUNTIME_ERROR


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
