from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass


@dataclass(slots=True)
class NvidiaHardwareStatus:
    available: bool
    executable: str | None
    gpu_names: tuple[str, ...] = ()
    error: str | None = None


def probe_nvidia_hardware() -> NvidiaHardwareStatus:
    executable = shutil.which("nvidia-smi")
    if executable is None:
        return NvidiaHardwareStatus(
            available=False,
            executable=None,
        )

    try:
        completed = subprocess.run(
            [
                executable,
                "--query-gpu=name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        return NvidiaHardwareStatus(
            available=False,
            executable=executable,
            error=str(exc),
        )

    gpu_names = tuple(
        line.strip()
        for line in completed.stdout.splitlines()
        if line.strip()
    )
    return NvidiaHardwareStatus(
        available=bool(gpu_names),
        executable=executable,
        gpu_names=gpu_names,
        error=None,
    )
