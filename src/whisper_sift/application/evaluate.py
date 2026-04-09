from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from whisper_sift.domain.evaluation import EvaluationReport, evaluate_golden_set
from whisper_sift.runtime.reporting import ProgressReporter, report_progress


@dataclass(slots=True)
class EvaluateRequest:
    golden_set_path: Path
    report_json_path: Path | None = None
    selected_cases: tuple[str, ...] = ()
    reporter: ProgressReporter | None = None


@dataclass(slots=True)
class EvaluateResult:
    report: EvaluationReport
    report_json_path: Path | None = None


def run_evaluate(request: EvaluateRequest) -> EvaluateResult:
    report = evaluate_golden_set(
        request.golden_set_path.expanduser(),
        selected_cases=request.selected_cases,
    )
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
    if report_json_path is not None:
        report_json_path.parent.mkdir(parents=True, exist_ok=True)
        report_json_path.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        report_progress(
            request.reporter,
            f"[eval] Report written to {report_json_path}",
        )

    return EvaluateResult(report=report, report_json_path=report_json_path)
