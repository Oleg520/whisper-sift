from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from dataclasses import dataclass

from whisper_sift.infrastructure.whisper_backend import is_fake_transcription_enabled
from whisper_sift.runtime.hardware import probe_nvidia_hardware


TORCH_PACKAGE_NAME = "torch"
TORCH_CPU_INDEX_URL = "https://download.pytorch.org/whl/cpu"
TORCH_CUDA_INDEX_URL = "https://download.pytorch.org/whl/cu126"

TRANSCRIPTION_DEPENDENCIES = {
    "torch": TORCH_PACKAGE_NAME,
    "whisper": "openai-whisper",
    "imageio_ffmpeg": "imageio-ffmpeg",
}

_dependencies_ready = False


@dataclass(slots=True, frozen=True)
class TorchInstallPlan:
    accelerator: str
    index_url: str


def collect_missing_transcription_packages() -> list[str]:
    if is_fake_transcription_enabled():
        return []

    system_ffmpeg_available = shutil.which("ffmpeg") is not None
    missing_packages: list[str] = []

    for module_name, package_name in TRANSCRIPTION_DEPENDENCIES.items():
        if module_name == "imageio_ffmpeg" and system_ffmpeg_available:
            continue
        if importlib.util.find_spec(module_name) is None:
            missing_packages.append(package_name)

    return missing_packages


def ensure_transcription_dependencies() -> None:
    global _dependencies_ready

    if _dependencies_ready:
        return

    if is_fake_transcription_enabled():
        _dependencies_ready = True
        return

    missing_packages = collect_missing_transcription_packages()
    if not missing_packages:
        _warn_if_gpu_is_available_but_torch_is_cpu_only()
        _dependencies_ready = True
        return

    install_commands = build_dependency_install_commands(missing_packages)
    if not install_commands:
        _dependencies_ready = True
        return

    print(
        "[bootstrap] Missing dependencies detected. Installing:",
        ", ".join(missing_packages),
    )
    for command in install_commands:
        print(f"[bootstrap] Running: {_format_command(command)}")
        subprocess.check_call(command)
    _warn_if_gpu_is_available_but_torch_is_cpu_only()
    _dependencies_ready = True


def build_dependency_install_commands(
    missing_packages: list[str] | None = None,
) -> list[list[str]]:
    missing_packages = (
        collect_missing_transcription_packages()
        if missing_packages is None
        else list(missing_packages)
    )
    if not missing_packages:
        return []

    commands: list[list[str]] = []
    remaining_packages = list(missing_packages)

    if TORCH_PACKAGE_NAME in remaining_packages:
        torch_plan = select_torch_install_plan()
        commands.append(build_torch_install_command(torch_plan))
        remaining_packages = [
            package_name
            for package_name in remaining_packages
            if package_name != TORCH_PACKAGE_NAME
        ]

    if remaining_packages:
        commands.append([sys.executable, "-m", "pip", "install", *remaining_packages])

    return commands


def select_torch_install_plan() -> TorchInstallPlan:
    nvidia = probe_nvidia_hardware()
    if nvidia.available:
        return TorchInstallPlan(
            accelerator="cuda",
            index_url=TORCH_CUDA_INDEX_URL,
        )

    return TorchInstallPlan(
        accelerator="cpu",
        index_url=TORCH_CPU_INDEX_URL,
    )


def build_torch_install_command(plan: TorchInstallPlan | None = None) -> list[str]:
    plan = select_torch_install_plan() if plan is None else plan
    return [
        sys.executable,
        "-m",
        "pip",
        "install",
        TORCH_PACKAGE_NAME,
        "--index-url",
        plan.index_url,
    ]


def detect_installed_torch_build() -> str | None:
    if importlib.util.find_spec("torch") is None:
        return None

    try:
        import torch
    except Exception:
        return None

    version = getattr(torch, "__version__", "") or ""
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)

    if cuda_version:
        return f"cu{str(cuda_version).replace('.', '')}"
    normalized_version = version.lower()
    if "+cu" in normalized_version:
        return normalized_version.rsplit("+", maxsplit=1)[-1]
    if "+cpu" in normalized_version:
        return "cpu"
    return "unknown"


def recommended_cuda_torch_command() -> str:
    return _format_command(
        build_torch_install_command(
            TorchInstallPlan("cuda", TORCH_CUDA_INDEX_URL)
        )
    )


def _warn_if_gpu_is_available_but_torch_is_cpu_only() -> None:
    nvidia = probe_nvidia_hardware()
    if not nvidia.available:
        return

    installed_build = detect_installed_torch_build()
    if installed_build != "cpu":
        return

    print(
        "[bootstrap] NVIDIA GPU detected, but installed torch build is CPU-only. "
        "Keeping the existing installation."
    )
    print(
        "[bootstrap] To enable CUDA for Whisper, reinstall torch with:",
        recommended_cuda_torch_command(),
    )


def _format_command(command: list[str]) -> str:
    return " ".join(command)
