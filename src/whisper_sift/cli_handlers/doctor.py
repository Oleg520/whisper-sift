from __future__ import annotations

import argparse

from whisper_sift.application.doctor import DoctorRequest, run_doctor
from whisper_sift.runtime.doctor import print_doctor_report


def handle_doctor(args: argparse.Namespace) -> int:
    report = run_doctor(DoctorRequest(install_missing=args.install_missing)).report
    print_doctor_report(report)
    return 0 if report.is_ready else 1
