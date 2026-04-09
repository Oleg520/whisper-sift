from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from whisper_sift.domain.evaluation import (
    EvaluationReport,
    EvaluationReportDiff,
    diff_evaluation_reports,
    evaluate_cases,
)
from whisper_sift.infrastructure.evaluation_store import (
    load_evaluation_report,
    load_golden_set,
    write_evaluation_diff,
    write_evaluation_report,
)
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


@dataclass(slots=True)
class EvaluateRequest:
    golden_set_path: Path
    report_json_path: Path | None = None
    baseline_report_path: Path | None = None
    diff_json_path: Path | None = None
    update_baseline: bool = False
    selected_cases: tuple[str, ...] = ()
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class EvaluateResult:
    report: EvaluationReport
    report_json_path: Path | None = None
    baseline_report_path: Path | None = None
    diff: EvaluationReportDiff | None = None
    diff_json_path: Path | None = None


def run_evaluate(request: EvaluateRequest) -> EvaluateResult:
    golden_set_path = request.golden_set_path.expanduser()
    cases = load_golden_set(
        golden_set_path,
        selected_cases=request.selected_cases,
    )
    report = evaluate_cases(cases, golden_set_path=golden_set_path)
    for case in report.cases:
        status = "PASS" if case.passed else "FAIL"
        report_progress(
            request.reporter,
            "[eval] "
            f"{case.case.name}: {status} "
            f"(required {len(case.matched_required)}/{len(case.case.required_questions)}, "
            f"forbidden {len(case.present_forbidden)}/{len(case.case.forbidden_questions)}, "
            f"extracted {len(case.extracted_questions)})",
        )

    report_progress(
        request.reporter,
        "[eval] Summary: "
        f"{report.passed_case_count}/{report.case_count} cases passed, "
        f"{report.required_matched}/{report.required_total} required matched, "
        f"{report.forbidden_present}/{report.forbidden_total} forbidden present.",
    )

    report_json_path = request.report_json_path.expanduser() if request.report_json_path else None
    baseline_report_path = (
        request.baseline_report_path.expanduser()
        if request.baseline_report_path
        else None
    )
    diff_json_path = request.diff_json_path.expanduser() if request.diff_json_path else None
    if report_json_path is not None:
        write_evaluation_report(report, report_json_path)
        report_progress(
            request.reporter,
            f"[eval] Report written to {report_json_path}",
        )

    diff: EvaluationReportDiff | None = None
    if baseline_report_path is not None:
        if not baseline_report_path.exists() and not request.update_baseline:
            raise RuntimeError(
                f"Baseline report not found: {baseline_report_path}"
            )
        if baseline_report_path.exists():
            baseline_report = load_evaluation_report(baseline_report_path)
            diff = diff_evaluation_reports(
                baseline_report,
                report,
                current_report_path=report_json_path,
                baseline_report_path=baseline_report_path,
            )
            _report_diff(diff, reporter=request.reporter)
            if diff_json_path is not None:
                write_evaluation_diff(diff, diff_json_path)
                report_progress(
                    request.reporter,
                    f"[eval] Diff written to {diff_json_path}",
                )

    if request.update_baseline and baseline_report_path is not None:
        write_evaluation_report(report, baseline_report_path)
        report_progress(
            request.reporter,
            f"[eval] Baseline updated at {baseline_report_path}",
        )

    return EvaluateResult(
        report=report,
        report_json_path=report_json_path,
        baseline_report_path=baseline_report_path,
        diff=diff,
        diff_json_path=diff_json_path,
    )


def _report_diff(
    diff: EvaluationReportDiff,
    *,
    reporter: ProgressReporter | None,
) -> None:
    for case in diff.case_diffs:
        if not case.has_changes:
            continue
        markers: list[str] = []
        if case.has_regression:
            markers.append("regression")
        if case.has_improvement:
            markers.append("improvement")
        if not markers:
            markers.append("changed")
        report_progress(
            reporter,
            "[eval:diff] "
            f"{case.name}: {', '.join(markers)} "
            f"(questions {case.baseline_extracted_question_count} -> {case.current_extracted_question_count})",
        )

    report_progress(
        reporter,
        "[eval:diff] Summary: "
        f"{diff.changed_case_count} changed, "
        f"{diff.regression_case_count} regressions, "
        f"{diff.improvement_case_count} improvements.",
    )
