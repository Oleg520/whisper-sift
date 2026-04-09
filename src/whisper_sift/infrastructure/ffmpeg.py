from __future__ import annotations

import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

from whisper_sift.paths import TOOLS_DIR


@dataclass(slots=True)
class FfmpegProbe:
    active_executable: Path | None
    source: str
    system_executable: Path | None
    bundled_executable: Path | None
    imageio_executable: Path | None


def ensure_ffmpeg_on_path() -> Path:
    system_ffmpeg = _resolve_system_ffmpeg()
    if system_ffmpeg is not None:
        return system_ffmpeg

    bundled_ffmpeg = _bundled_ffmpeg_alias()
    if bundled_ffmpeg is not None:
        if os.name != "nt":
            _ensure_executable(bundled_ffmpeg)
        _prepend_to_path(bundled_ffmpeg.parent)
        return bundled_ffmpeg

    ffmpeg_source = _resolve_imageio_ffmpeg()
    if ffmpeg_source is None:
        raise RuntimeError(
            "FFmpeg is unavailable. Install ffmpeg system-wide or install "
            "the 'imageio-ffmpeg' package."
        )

    ffmpeg_dir = TOOLS_DIR / "ffmpeg"
    ffmpeg_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_alias = ffmpeg_dir / _ffmpeg_alias_name()
    if _needs_refresh(ffmpeg_alias, ffmpeg_source):
        shutil.copy2(ffmpeg_source, ffmpeg_alias)
    if os.name != "nt":
        _ensure_executable(ffmpeg_alias)
    _prepend_to_path(ffmpeg_dir)

    return ffmpeg_alias


def probe_ffmpeg_environment() -> FfmpegProbe:
    system_ffmpeg = _resolve_system_ffmpeg()
    bundled_executable = _bundled_ffmpeg_alias()
    imageio_executable = _resolve_imageio_ffmpeg()

    if system_ffmpeg is not None:
        return FfmpegProbe(
            active_executable=system_ffmpeg,
            source="system",
            system_executable=system_ffmpeg,
            bundled_executable=bundled_executable,
            imageio_executable=imageio_executable,
        )

    if bundled_executable is not None:
        return FfmpegProbe(
            active_executable=bundled_executable,
            source="bundled",
            system_executable=None,
            bundled_executable=bundled_executable,
            imageio_executable=imageio_executable,
        )

    if imageio_executable is not None:
        return FfmpegProbe(
            active_executable=imageio_executable,
            source="imageio-ffmpeg",
            system_executable=None,
            bundled_executable=None,
            imageio_executable=imageio_executable,
        )

    return FfmpegProbe(
        active_executable=None,
        source="missing",
        system_executable=None,
        bundled_executable=None,
        imageio_executable=None,
    )


def _resolve_system_ffmpeg() -> Path | None:
    resolved = shutil.which("ffmpeg")
    if resolved is None:
        return None
    return Path(resolved).resolve()


def _resolve_imageio_ffmpeg() -> Path | None:
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError:
        return None

    return Path(imageio_ffmpeg.get_ffmpeg_exe()).resolve()


def _bundled_ffmpeg_alias() -> Path | None:
    ffmpeg_alias = TOOLS_DIR / "ffmpeg" / _ffmpeg_alias_name()
    if not ffmpeg_alias.exists():
        return None
    return ffmpeg_alias.resolve()


def _ffmpeg_alias_name() -> str:
    return "ffmpeg.exe" if os.name == "nt" else "ffmpeg"


def _prepend_to_path(directory: Path) -> None:
    current_path = os.environ.get("PATH", "")
    entries = current_path.split(os.pathsep) if current_path else []
    if str(directory) not in entries:
        os.environ["PATH"] = str(directory) + os.pathsep + current_path if current_path else str(directory)


def _needs_refresh(target: Path, source: Path) -> bool:
    if not target.exists():
        return True
    target_stat = target.stat()
    source_stat = source.stat()
    return (
        target_stat.st_size != source_stat.st_size
        or target_stat.st_mtime_ns != source_stat.st_mtime_ns
    )


def _ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(
        mode
        | stat.S_IXUSR
        | stat.S_IXGRP
        | stat.S_IXOTH
    )
