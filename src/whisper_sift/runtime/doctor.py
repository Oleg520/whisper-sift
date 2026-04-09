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
    recommended_cuda_torch_command,
)
from whisper_sift.runtime.hardware import NvidiaHardwareStatus, probe_nvidia_hardware


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
    build_variant: str | None = None
    cuda_version: str | None = None


@dataclass(slots=True)
class DoctorReport:
    python_executable: str
    python_version: str
    package_root: Path
    runtime_root: Path
    project_root: Path | None
    platform_name: str
    nvidia: NvidiaHardwareStatus
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
    def warnings(self) -> tuple[str, ...]:
        warnings: list[str] = []

        if self.nvidia.error:
            warnings.append(
                f"NVIDIA tooling detected, but GPU probing failed: {self.nvidia.error}"
            )

        if self.nvidia.available and not self.torch.available:
            warnings.append(
                "NVIDIA GPU detected, but torch is not installed yet. "
                "The first bootstrap will install a CUDA-enabled torch build."
            )
        elif self.nvidia.available and self.torch.available:
            if self.torch.build_variant == "cpu":
                warnings.append(
                    "NVIDIA GPU detected, but installed torch build is CPU-only. "
                    f"Recommended command: {recommended_cuda_torch_command()}"
                )
            elif (
                self.torch.build_variant
                and self.torch.build_variant.startswith("cu")
                and not self.torch.cuda_available
            ):
                warnings.append(
                    "CUDA-enabled torch build is installed, but torch.cuda.is_available() "
                    "is false. Check the NVIDIA driver and active Python environment."
                )

        return tuple(warnings)

    @property
    def is_ready(self) -> bool:
        return not self.issues


def collect_doctor_report(*, install_missing: bool = False) -> DoctorReport:
    if install_missing:
        ensure_transcription_dependencies()

    ffmpeg = probe_ffmpeg_environment()
    nvidia = probe_nvidia_hardware()
    dependencies = _collect_dependency_statuses(system_ffmpeg_available=ffmpeg.system_executable is not None)
    torch = _collect_torch_status()

    return DoctorReport(
        python_executable=sys.executable,
        python_version=platform.python_version(),
        package_root=PACKAGE_ROOT,
        runtime_root=RUNTIME_ROOT,
        project_root=PROJECT_ROOT,
        platform_name=platform.platform(),
        nvidia=nvidia,
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
    if report.nvidia.available:
        print(
            "[nvidia]   detected"
            f" ({', '.join(report.nvidia.gpu_names)})"
        )
        if report.nvidia.executable is not None:
            print(f"[nvidia]   executable={report.nvidia.executable}")
    elif report.nvidia.executable is not None:
        print("[nvidia]   unavailable")
        print(f"[nvidia]   executable={report.nvidia.executable}")
    else:
        print("[nvidia]   not found")

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
        build = report.torch.build_variant or "unknown"
        print(
            f"[torch]    version={report.torch.version or 'unknown'} build={build}"
        )
        if report.torch.cuda_version is not None:
            print(f"[torch]    cuda_version={report.torch.cuda_version}")
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

    for warning in report.warnings:
        print(f"[warn]     {warning}")

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
            build_variant=None,
            cuda_version=None,
            cuda_available=False,
            mps_available=False,
        )

    try:
        import torch
    except Exception:
        return TorchStatus(
            available=False,
            version=None,
            build_variant=None,
            cuda_version=None,
            cuda_available=False,
            mps_available=False,
        )

    version = getattr(torch, "__version__", None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    build_variant = _detect_torch_build_variant(version, cuda_version)
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    return TorchStatus(
        available=True,
        version=version,
        build_variant=build_variant,
        cuda_version=str(cuda_version) if cuda_version is not None else None,
        cuda_available=bool(torch.cuda.is_available()),
        mps_available=mps_available,
    )


def _detect_torch_build_variant(
    version: str | None,
    cuda_version: str | None,
) -> str | None:
    if cuda_version:
        return f"cu{str(cuda_version).replace('.', '')}"

    normalized_version = (version or "").lower()
    if "+cu" in normalized_version:
        return normalized_version.rsplit("+", maxsplit=1)[-1]
    if "+cpu" in normalized_version:
        return "cpu"

    return None
