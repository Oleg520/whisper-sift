from __future__ import annotations

import json
from pathlib import Path

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
)
from whisper_sift.domain.evaluation import (
    CaseEvaluation,
    EvaluationReport,
    EvaluationReportDiff,
    GoldenCase,
)
from whisper_sift.infrastructure.filesystem import write_json_file


def load_golden_set(
    golden_set_path: Path,
    *,
    selected_cases: tuple[str, ...] = (),
) -> tuple[GoldenCase, ...]:
    payload = json.loads(golden_set_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise RuntimeError("Golden set must contain a non-empty 'cases' list.")

    allowed_names = {name.strip() for name in selected_cases if name.strip()}
    manifest_dir = golden_set_path.parent
    cases: list[GoldenCase] = []

    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise RuntimeError(f"Golden set case #{index} must be an object.")

        name = str(raw_case.get("name") or "").strip()
        source = str(raw_case.get("source") or "").strip()
        if not name or not source:
            raise RuntimeError(
                f"Golden set case #{index} must define both 'name' and 'source'."
            )
        if allowed_names and name not in allowed_names:
            continue

        source_path = Path(source)
        if not source_path.is_absolute():
            source_path = manifest_dir / source_path
        resolved_source_path = source_path.resolve()

        cases.append(
            GoldenCase(
                name=name,
                source_path=resolved_source_path,
                transcript_text=resolved_source_path.read_text(encoding="utf-8"),
                required_questions=_normalize_string_tuple(raw_case.get("required_questions")),
                forbidden_questions=_normalize_string_tuple(raw_case.get("forbidden_questions")),
                interviewer_labels=_normalize_string_tuple(raw_case.get("interviewer_labels")),
                deduplicate=bool(
                    raw_case.get("deduplicate", DEFAULT_DEDUPLICATE_QUESTIONS)
                ),
                min_length=int(raw_case.get("min_length", DEFAULT_MIN_QUESTION_LENGTH)),
                max_length=int(raw_case.get("max_length", DEFAULT_MAX_QUESTION_LENGTH)),
            )
        )

    if allowed_names and not cases:
        available = ", ".join(
            str(case.get("name", "")) for case in raw_cases if isinstance(case, dict)
        )
        raise RuntimeError(
            "None of the requested evaluation cases were found in the golden set. "
            f"Available cases: {available}"
        )

    return tuple(cases)


def load_evaluation_report(report_path: Path) -> EvaluationReport:
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list):
        raise RuntimeError("Evaluation report must contain a 'cases' list.")

    cases: list[CaseEvaluation] = []
    golden_set_path = Path(str(payload.get("golden_set_path") or report_path))
    for index, raw_case in enumerate(raw_cases, start=1):
        if not isinstance(raw_case, dict):
            raise RuntimeError(f"Evaluation report case #{index} must be an object.")

        case_name = str(raw_case.get("name") or "").strip()
        source_path = Path(str(raw_case.get("source_path") or ""))
        matched_required = _normalize_string_tuple(raw_case.get("matched_required"))
        missing_required = _normalize_string_tuple(raw_case.get("missing_required"))
        present_forbidden = _normalize_string_tuple(raw_case.get("present_forbidden"))
        extracted_questions = _normalize_string_tuple(raw_case.get("extracted_questions"))

        cases.append(
            CaseEvaluation(
                case=GoldenCase(
                    name=case_name,
                    source_path=source_path,
                    transcript_text="",
                    required_questions=tuple(matched_required + missing_required),
                    forbidden_questions=tuple(present_forbidden),
                ),
                extracted_questions=tuple(extracted_questions),
                matched_required=tuple(matched_required),
                missing_required=tuple(missing_required),
                present_forbidden=tuple(present_forbidden),
            )
        )

    return EvaluationReport(
        golden_set_path=golden_set_path,
        cases=tuple(cases),
    )


def write_evaluation_report(report: EvaluationReport, path: Path) -> Path:
    return write_json_file(path, report.to_dict())


def write_evaluation_diff(diff: EvaluationReportDiff, path: Path) -> Path:
    return write_json_file(path, diff.to_dict())


def _normalize_string_tuple(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise RuntimeError("Golden set values must be lists of strings.")
    result: list[str] = []
    for item in value:
        text = str(item).strip()
        if text:
            result.append(text)
    return tuple(result)
