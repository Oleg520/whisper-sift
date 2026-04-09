from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    transcript_text: str = ""
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


@dataclass(slots=True, frozen=True)
class CaseEvaluationDiff:
    name: str
    baseline_passed: bool
    current_passed: bool
    baseline_extracted_question_count: int
    current_extracted_question_count: int
    newly_missing_required: tuple[str, ...]
    resolved_missing_required: tuple[str, ...]
    newly_present_forbidden: tuple[str, ...]
    resolved_forbidden: tuple[str, ...]
    added_questions: tuple[str, ...]
    removed_questions: tuple[str, ...]

    @property
    def has_changes(self) -> bool:
        return any(
            (
                self.baseline_passed != self.current_passed,
                self.baseline_extracted_question_count != self.current_extracted_question_count,
                self.newly_missing_required,
                self.resolved_missing_required,
                self.newly_present_forbidden,
                self.resolved_forbidden,
                self.added_questions,
                self.removed_questions,
            )
        )

    @property
    def has_regression(self) -> bool:
        return bool(self.newly_missing_required or self.newly_present_forbidden)

    @property
    def has_improvement(self) -> bool:
        return bool(self.resolved_missing_required or self.resolved_forbidden)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "baseline_passed": self.baseline_passed,
            "current_passed": self.current_passed,
            "baseline_extracted_question_count": self.baseline_extracted_question_count,
            "current_extracted_question_count": self.current_extracted_question_count,
            "extracted_question_count_delta": (
                self.current_extracted_question_count - self.baseline_extracted_question_count
            ),
            "newly_missing_required": list(self.newly_missing_required),
            "resolved_missing_required": list(self.resolved_missing_required),
            "newly_present_forbidden": list(self.newly_present_forbidden),
            "resolved_forbidden": list(self.resolved_forbidden),
            "added_questions": list(self.added_questions),
            "removed_questions": list(self.removed_questions),
            "has_changes": self.has_changes,
            "has_regression": self.has_regression,
            "has_improvement": self.has_improvement,
        }


@dataclass(slots=True, frozen=True)
class EvaluationReportDiff:
    baseline_report_path: Path
    current_report_path: Path | None
    case_diffs: tuple[CaseEvaluationDiff, ...]

    @property
    def changed_case_count(self) -> int:
        return sum(1 for case in self.case_diffs if case.has_changes)

    @property
    def regression_case_count(self) -> int:
        return sum(1 for case in self.case_diffs if case.has_regression)

    @property
    def improvement_case_count(self) -> int:
        return sum(1 for case in self.case_diffs if case.has_improvement)

    @property
    def has_changes(self) -> bool:
        return self.changed_case_count > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_report_path": str(self.baseline_report_path),
            "current_report_path": str(self.current_report_path) if self.current_report_path else None,
            "changed_case_count": self.changed_case_count,
            "regression_case_count": self.regression_case_count,
            "improvement_case_count": self.improvement_case_count,
            "has_changes": self.has_changes,
            "cases": [case.to_dict() for case in self.case_diffs],
        }

def evaluate_cases(
    cases: tuple[GoldenCase, ...],
    *,
    golden_set_path: Path,
) -> EvaluationReport:
    if not cases:
        raise RuntimeError("Golden set does not contain any evaluation cases.")

    evaluations = tuple(_evaluate_case(case) for case in cases)
    return EvaluationReport(
        golden_set_path=golden_set_path,
        cases=evaluations,
    )


def diff_evaluation_reports(
    baseline_report: EvaluationReport,
    current_report: EvaluationReport,
    *,
    current_report_path: Path | None = None,
    baseline_report_path: Path | None = None,
) -> EvaluationReportDiff:
    baseline_cases = {case.case.name: case for case in baseline_report.cases}
    current_cases = {case.case.name: case for case in current_report.cases}
    all_case_names = sorted(set(baseline_cases) | set(current_cases))

    diffs: list[CaseEvaluationDiff] = []
    for case_name in all_case_names:
        baseline_case = baseline_cases.get(case_name)
        current_case = current_cases.get(case_name)

        baseline_missing = {
            _canonicalize_question(question): question
            for question in (baseline_case.missing_required if baseline_case else ())
        }
        current_missing = {
            _canonicalize_question(question): question
            for question in (current_case.missing_required if current_case else ())
        }
        baseline_forbidden = {
            _canonicalize_question(question): question
            for question in (baseline_case.present_forbidden if baseline_case else ())
        }
        current_forbidden = {
            _canonicalize_question(question): question
            for question in (current_case.present_forbidden if current_case else ())
        }

        baseline_questions = {
            _canonicalize_question(question): question
            for question in (baseline_case.extracted_questions if baseline_case else ())
        }
        current_questions = {
            _canonicalize_question(question): question
            for question in (current_case.extracted_questions if current_case else ())
        }

        diffs.append(
            CaseEvaluationDiff(
                name=case_name,
                baseline_passed=baseline_case.passed if baseline_case else False,
                current_passed=current_case.passed if current_case else False,
                baseline_extracted_question_count=len(baseline_questions),
                current_extracted_question_count=len(current_questions),
                newly_missing_required=tuple(
                    current_missing[key]
                    for key in sorted(current_missing.keys() - baseline_missing.keys())
                ),
                resolved_missing_required=tuple(
                    baseline_missing[key]
                    for key in sorted(baseline_missing.keys() - current_missing.keys())
                ),
                newly_present_forbidden=tuple(
                    current_forbidden[key]
                    for key in sorted(current_forbidden.keys() - baseline_forbidden.keys())
                ),
                resolved_forbidden=tuple(
                    baseline_forbidden[key]
                    for key in sorted(baseline_forbidden.keys() - current_forbidden.keys())
                ),
                added_questions=tuple(
                    current_questions[key]
                    for key in sorted(current_questions.keys() - baseline_questions.keys())
                ),
                removed_questions=tuple(
                    baseline_questions[key]
                    for key in sorted(baseline_questions.keys() - current_questions.keys())
                ),
            )
        )

    return EvaluationReportDiff(
        baseline_report_path=baseline_report_path or baseline_report.golden_set_path,
        current_report_path=current_report_path,
        case_diffs=tuple(diffs),
    )


def _evaluate_case(case: GoldenCase) -> CaseEvaluation:
    extraction_result = extract_questions(
        case.transcript_text,
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
def _canonicalize_question(value: str) -> str:
    normalized = CANONICAL_QUESTION_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")
