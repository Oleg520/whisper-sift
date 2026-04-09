from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json
import re

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
)
from whisper_sift.domain.extraction import extract_questions


CANONICAL_QUESTION_RE = re.compile(r"\s+")


@dataclass(slots=True, frozen=True)
class GoldenCase:
    name: str
    source_path: Path
    required_questions: tuple[str, ...] = ()
    forbidden_questions: tuple[str, ...] = ()
    interviewer_labels: tuple[str, ...] = ()
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH


@dataclass(slots=True, frozen=True)
class CaseEvaluation:
    case: GoldenCase
    extracted_questions: tuple[str, ...]
    matched_required: tuple[str, ...]
    missing_required: tuple[str, ...]
    present_forbidden: tuple[str, ...]

    @property
    def passed(self) -> bool:
        return not self.missing_required and not self.present_forbidden

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.case.name,
            "source_path": str(self.case.source_path),
            "passed": self.passed,
            "required_total": len(self.case.required_questions),
            "required_matched": len(self.matched_required),
            "forbidden_total": len(self.case.forbidden_questions),
            "forbidden_present": len(self.present_forbidden),
            "matched_required": list(self.matched_required),
            "missing_required": list(self.missing_required),
            "present_forbidden": list(self.present_forbidden),
            "extracted_question_count": len(self.extracted_questions),
            "extracted_questions": list(self.extracted_questions),
        }


@dataclass(slots=True, frozen=True)
class EvaluationReport:
    golden_set_path: Path
    cases: tuple[CaseEvaluation, ...]

    @property
    def case_count(self) -> int:
        return len(self.cases)

    @property
    def passed_case_count(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed_case_count(self) -> int:
        return self.case_count - self.passed_case_count

    @property
    def required_total(self) -> int:
        return sum(len(case.case.required_questions) for case in self.cases)

    @property
    def required_matched(self) -> int:
        return sum(len(case.matched_required) for case in self.cases)

    @property
    def forbidden_total(self) -> int:
        return sum(len(case.case.forbidden_questions) for case in self.cases)

    @property
    def forbidden_present(self) -> int:
        return sum(len(case.present_forbidden) for case in self.cases)

    @property
    def is_passing(self) -> bool:
        return self.failed_case_count == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "golden_set_path": str(self.golden_set_path),
            "case_count": self.case_count,
            "passed_case_count": self.passed_case_count,
            "failed_case_count": self.failed_case_count,
            "required_total": self.required_total,
            "required_matched": self.required_matched,
            "forbidden_total": self.forbidden_total,
            "forbidden_present": self.forbidden_present,
            "is_passing": self.is_passing,
            "cases": [case.to_dict() for case in self.cases],
        }


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

        cases.append(
            GoldenCase(
                name=name,
                source_path=source_path,
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
        available = ", ".join(str(case.get("name", "")) for case in raw_cases if isinstance(case, dict))
        raise RuntimeError(
            "None of the requested evaluation cases were found in the golden set. "
            f"Available cases: {available}"
        )

    return tuple(cases)


def evaluate_golden_set(
    golden_set_path: Path,
    *,
    selected_cases: tuple[str, ...] = (),
) -> EvaluationReport:
    cases = load_golden_set(golden_set_path, selected_cases=selected_cases)
    if not cases:
        raise RuntimeError("Golden set does not contain any evaluation cases.")

    evaluations = tuple(_evaluate_case(case) for case in cases)
    return EvaluationReport(
        golden_set_path=golden_set_path,
        cases=evaluations,
    )


def _evaluate_case(case: GoldenCase) -> CaseEvaluation:
    transcript_text = case.source_path.read_text(encoding="utf-8")
    extraction_result = extract_questions(
        transcript_text,
        deduplicate=case.deduplicate,
        min_length=case.min_length,
        max_length=case.max_length,
        interviewer_labels=case.interviewer_labels,
        source_name=case.source_path.name,
    )
    extracted_questions = extraction_result.question_texts
    extracted_canonical = {
        _canonicalize_question(question): question for question in extracted_questions
    }

    matched_required: list[str] = []
    missing_required: list[str] = []
    for required in case.required_questions:
        if _canonicalize_question(required) in extracted_canonical:
            matched_required.append(required)
        else:
            missing_required.append(required)

    present_forbidden = [
        forbidden
        for forbidden in case.forbidden_questions
        if _canonicalize_question(forbidden) in extracted_canonical
    ]

    return CaseEvaluation(
        case=case,
        extracted_questions=tuple(extracted_questions),
        matched_required=tuple(matched_required),
        missing_required=tuple(missing_required),
        present_forbidden=tuple(present_forbidden),
    )


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


def _canonicalize_question(value: str) -> str:
    normalized = CANONICAL_QUESTION_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")
