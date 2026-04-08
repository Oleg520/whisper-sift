from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


TRANSCRIPTION_DEPENDENCIES = {
    "torch": "torch",
    "whisper": "openai-whisper",
    "imageio_ffmpeg": "imageio-ffmpeg",
}

_dependencies_ready = False


def collect_missing_transcription_packages() -> list[str]:
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

    missing_packages = collect_missing_transcription_packages()
    if not missing_packages:
        _dependencies_ready = True
        return

    print(
        "[bootstrap] Missing dependencies detected. Installing:",
        ", ".join(missing_packages),
    )
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", *missing_packages]
    )
    _dependencies_ready = True
