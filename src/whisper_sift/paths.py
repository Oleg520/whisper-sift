from __future__ import annotations

import os
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PACKAGE_ROOT.parent
APP_NAME = "whisper-sift"


def _detect_project_root() -> Path | None:
    candidate = SRC_ROOT.parent
    if (candidate / "pyproject.toml").exists():
        return candidate
    return None


def _default_runtime_root() -> Path:
    if sys.platform == "win32":
        base_dir = (
            os.environ.get("LOCALAPPDATA")
            or os.environ.get("APPDATA")
            or str(Path.home() / "AppData" / "Local")
        )
    elif sys.platform == "darwin":
        base_dir = str(Path.home() / "Library" / "Caches")
    else:
        base_dir = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")

    return Path(base_dir) / APP_NAME


def _resolve_runtime_root() -> Path:
    override = os.environ.get("WHISPER_SIFT_RUNTIME_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return _default_runtime_root().resolve()


PROJECT_ROOT = _detect_project_root()
RUNTIME_ROOT = _resolve_runtime_root()
TOOLS_DIR = RUNTIME_ROOT / "tools"
