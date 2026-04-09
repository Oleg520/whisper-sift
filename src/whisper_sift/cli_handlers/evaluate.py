from __future__ import annotations

import argparse
from pathlib import Path

from whisper_sift.application.evaluate import EvaluateRequest, run_evaluate
from whisper_sift.config import EvaluationPolicy
from whisper_sift.runtime.reporting import ConsoleReporter


def handle_evaluate(args: argparse.Namespace) -> int:
    reporter = ConsoleReporter()
    golden_set_path = args.golden_set.expanduser()
    report_json_path = (
        args.report_json.expanduser()
        if args.report_json
        else golden_set_path.with_name("latest_report.json")
    )
    if args.diff_json and not (args.baseline_report or args.update_baseline):
        raise RuntimeError("--diff-json requires --baseline-report or --update-baseline.")

    baseline_report_path: Path | None
    if args.baseline_report or args.update_baseline:
        baseline_report_path = (
            args.baseline_report.expanduser()
            if args.baseline_report
            else golden_set_path.with_name("baseline_report.json")
        )
    else:
        baseline_report_path = None

    if baseline_report_path is not None:
        diff_json_path = (
            args.diff_json.expanduser()
            if args.diff_json
            else golden_set_path.with_name("latest_diff.json")
        )
    else:
        diff_json_path = None

    result = run_evaluate(
        EvaluateRequest(
            golden_set_path=golden_set_path,
            report_json_path=report_json_path,
            baseline_report_path=baseline_report_path,
            diff_json_path=diff_json_path,
            evaluation=EvaluationPolicy(
                selected_cases=tuple(args.case),
                update_baseline=args.update_baseline,
            ),
            reporter=reporter,
        )
    )
    return 0 if result.report.is_passing else 1
