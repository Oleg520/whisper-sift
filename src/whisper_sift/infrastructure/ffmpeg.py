from __future__ import annotations

import os
import shutil
from pathlib import Path

import imageio_ffmpeg

from whisper_sift.paths import TOOLS_DIR


def ensure_ffmpeg_on_path() -> Path:
    ffmpeg_source = Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()
    ffmpeg_dir = TOOLS_DIR / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_alias = ffmpeg_dir / "ffmpeg.exe"
    if not ffmpeg_alias.exists():
        shutil.copy2(ffmpeg_source, ffmpeg_alias)

    current_path = os.environ.get("PATH", "")
    if str(ffmpeg_dir) not in current_path.split(os.pathsep):
        os.environ["PATH"] = str(ffmpeg_dir) + os.pathsep + current_path

    return ffmpeg_alias
