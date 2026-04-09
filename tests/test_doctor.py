from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.infrastructure.ffmpeg import FfmpegProbe
from whisper_sift.runtime.doctor import (
    DoctorReport,
    TorchStatus,
    _collect_dependency_statuses,
)


class DoctorTests(unittest.TestCase):
    @patch("whisper_sift.runtime.doctor.importlib.util.find_spec")
    def test_imageio_dependency_becomes_optional_with_system_ffmpeg(
        self,
        find_spec_mock,
    ) -> None:
        def fake_find_spec(module_name: str) -> object | None:
            if module_name in {"torch", "whisper"}:
                return object()
            return None

        find_spec_mock.side_effect = fake_find_spec

        statuses = _collect_dependency_statuses(system_ffmpeg_available=True)

        imageio_status = next(
            status for status in statuses if status.module_name == "imageio_ffmpeg"
        )
        self.assertFalse(imageio_status.available)
        self.assertFalse(imageio_status.required)

    def test_report_can_be_ready_without_imageio_when_system_ffmpeg_exists(self) -> None:
        report = DoctorReport(
            python_executable=sys.executable,
            python_version="3.11.0",
            package_root=PROJECT_ROOT / "src" / "whisper_sift",
            runtime_root=Path("/tmp/whisper-sift"),
            project_root=PROJECT_ROOT,
            platform_name="TestOS",
            dependencies=(
                _status("torch", "torch", True, True),
                _status("whisper", "openai-whisper", True, True),
                _status("imageio_ffmpeg", "imageio-ffmpeg", False, False),
            ),
            torch=TorchStatus(
                available=True,
                version="1.0.0",
                cuda_available=False,
                mps_available=False,
            ),
            ffmpeg=FfmpegProbe(
                active_executable=Path("/usr/bin/ffmpeg"),
                source="system",
                system_executable=Path("/usr/bin/ffmpeg"),
                bundled_executable=None,
                imageio_executable=None,
            ),
        )

        self.assertTrue(report.is_ready)
        self.assertEqual((), report.issues)


def _status(module_name: str, package_name: str, available: bool, required: bool):
    from whisper_sift.runtime.doctor import DependencyStatus

    return DependencyStatus(
        module_name=module_name,
        package_name=package_name,
        available=available,
        required=required,
        detail=None,
    )


if __name__ == "__main__":
    unittest.main()
