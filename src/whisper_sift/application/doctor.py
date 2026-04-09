from __future__ import annotations

from dataclasses import dataclass

from whisper_sift.runtime.doctor import DoctorReport


@dataclass(slots=True)
class DoctorRequest:
    install_missing: bool = False


@dataclass(slots=True)
class DoctorResult:
    report: DoctorReport


def run_doctor(request: DoctorRequest) -> DoctorResult:
    from whisper_sift.runtime.doctor import collect_doctor_report

    report = collect_doctor_report(install_missing=request.install_missing)
    return DoctorResult(report=report)

