from __future__ import annotations

import importlib.util
import subprocess
import sys


TRANSCRIPTION_DEPENDENCIES = {
    "torch": "torch",
    "whisper": "openai-whisper",
    "imageio_ffmpeg": "imageio-ffmpeg",
}

_dependencies_ready = False


def ensure_transcription_dependencies() -> None:
    global _dependencies_ready

    if _dependencies_ready:
        return

    missing_packages = [
        package_name
        for module_name, package_name in TRANSCRIPTION_DEPENDENCIES.items()
        if importlib.util.find_spec(module_name) is None
    ]

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
