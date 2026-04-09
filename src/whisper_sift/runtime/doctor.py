from __future__ import annotations

import importlib.util
import platform
import sys
from dataclasses import dataclass
from pathlib import Path

from whisper_sift.infrastructure.ffmpeg import FfmpegProbe, probe_ffmpeg_environment
from whisper_sift.paths import PACKAGE_ROOT, PROJECT_ROOT, RUNTIME_ROOT
from whisper_sift.runtime.dependencies import (
    TRANSCRIPTION_DEPENDENCIES,
    ensure_transcription_dependencies,
)


@dataclass(slots=True)
class DependencyStatus:
    module_name: str
    package_name: str
    available: bool
    required: bool
    detail: str | None = None


@dataclass(slots=True)
class TorchStatus:
    available: bool
    version: str | None
    cuda_available: bool
    mps_available: bool


@dataclass(slots=True)
class DoctorReport:
    python_executable: str
    python_version: str
    package_root: Path
    runtime_root: Path
    project_root: Path | None
    platform_name: str
    dependencies: tuple[DependencyStatus, ...]
    torch: TorchStatus
    ffmpeg: FfmpegProbe

    @property
    def issues(self) -> tuple[str, ...]:
        issues: list[str] = []

        for dependency in self.dependencies:
            if dependency.required and not dependency.available:
                issues.append(
                    f"Missing dependency: {dependency.package_name} "
                    f"(module '{dependency.module_name}')"
                )

        if self.ffmpeg.active_executable is None:
            issues.append("FFmpeg not found via PATH or imageio-ffmpeg.")

        return tuple(issues)

    @property
    def is_ready(self) -> bool:
        return not self.issues


def collect_doctor_report(*, install_missing: bool = False) -> DoctorReport:
    if install_missing:
        ensure_transcription_dependencies()

    ffmpeg = probe_ffmpeg_environment()
    dependencies = _collect_dependency_statuses(system_ffmpeg_available=ffmpeg.system_executable is not None)
    torch = _collect_torch_status()

    return DoctorReport(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        package_root=PACKAGE_ROOT,
        runtime_root=RUNTIME_ROOT,
        project_root=PROJECT_ROOT,
        platform_name=platform.platform(),
        dependencies=dependencies,
        torch=torch,
        ffmpeg=ffmpeg,
    )


def print_doctor_report(report: DoctorReport) -> None:
    print(f"[python]   {report.python_version} ({report.python_executable})")
    if report.project_root is not None:
        print(f"[project]  {report.project_root}")
    print(f"[package]  {report.package_root}")
    print(f"[runtime]  {report.runtime_root}")
    print(f"[platform] {report.platform_name}")

    for dependency in report.dependencies:
        if dependency.available:
            state = "ok"
        elif dependency.required:
            state = "missing"
        else:
            state = "optional"

        location = dependency.detail or "-"
        print(
            f"[dep]      {dependency.module_name:<14} "
            f"state={state:<8} package={dependency.package_name} location={location}"
        )

    if report.torch.available:
        print(f"[torch]    version={report.torch.version or 'unknown'}")
        print(f"[cuda]     {'available' if report.torch.cuda_available else 'unavailable'}")
        print(f"[mps]      {'available' if report.torch.mps_available else 'unavailable'}")
    else:
        print("[torch]    unavailable")
        print("[cuda]     unavailable")
        print("[mps]      unavailable")

    if report.ffmpeg.active_executable is None:
        print("[ffmpeg]   unavailable")
    else:
        print(
            f"[ffmpeg]   source={report.ffmpeg.source} "
            f"active={report.ffmpeg.active_executable}"
        )
        if report.ffmpeg.system_executable is not None:
            print(f"[ffmpeg]   system={report.ffmpeg.system_executable}")
        if report.ffmpeg.bundled_executable is not None:
            print(f"[ffmpeg]   bundled={report.ffmpeg.bundled_executable}")
        if report.ffmpeg.imageio_executable is not None:
            print(f"[ffmpeg]   imageio={report.ffmpeg.imageio_executable}")

    if report.is_ready:
        print("[status]   ready for transcription")
        return

    print("[status]   issues detected")
    for issue in report.issues:
        print(f"[issue]    {issue}")


def _collect_dependency_statuses(*, system_ffmpeg_available: bool) -> tuple[DependencyStatus, ...]:
    statuses: list[DependencyStatus] = []

    for module_name, package_name in TRANSCRIPTION_DEPENDENCIES.items():
        required = not (module_name == "imageio_ffmpeg" and system_ffmpeg_available)
        spec = importlib.util.find_spec(module_name)
        statuses.append(
            DependencyStatus(
                module_name=module_name,
                package_name=package_name,
                available=spec is not None,
                required=required,
                detail=getattr(spec, "origin", None) if spec is not None else None,
            )
        )

    return tuple(statuses)


def _collect_torch_status() -> TorchStatus:
    if importlib.util.find_spec("torch") is None:
        return TorchStatus(
            available=False,
            version=None,
            cuda_available=False,
            mps_available=False,
        )

    try:
        import torch
    except Exception:
        return TorchStatus(
            available=False,
            version=None,
            cuda_available=False,
            mps_available=False,
        )

    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return TorchStatus(
        available=True,
        version=getattr(torch, "__version__", None),
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=mps_available,
    )
