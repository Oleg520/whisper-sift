from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.runtime.dependencies import (
    TORCH_CPU_INDEX_URL,
    TORCH_CUDA_INDEX_URL,
    build_dependency_install_commands,
    build_torch_install_command,
    collect_missing_transcription_packages,
)
from whisper_sift.runtime.hardware import NvidiaHardwareStatus


class RuntimeDependencyTests(unittest.TestCase):
    @patch("whisper_sift.runtime.dependencies.is_fake_transcription_enabled")
    def test_fake_backend_makes_runtime_dependencies_optional(self, fake_enabled_mock) -> None:
        fake_enabled_mock.return_value = True

        missing_packages = collect_missing_transcription_packages()

        self.assertEqual([], missing_packages)

    @patch("whisper_sift.runtime.dependencies.shutil.which")
    @patch("whisper_sift.runtime.dependencies.importlib.util.find_spec")
    def test_system_ffmpeg_makes_imageio_optional(
        self,
        find_spec_mock,
        which_mock,
    ) -> None:
        which_mock.return_value = "/usr/bin/ffmpeg"

        def fake_find_spec(module_name: str) -> object | None:
            if module_name in {"torch", "whisper"}:
                return object()
            return None

        find_spec_mock.side_effect = fake_find_spec

        missing_packages = collect_missing_transcription_packages()

        self.assertEqual([], missing_packages)

    @patch("whisper_sift.runtime.dependencies.shutil.which")
    @patch("whisper_sift.runtime.dependencies.importlib.util.find_spec")
    def test_imageio_is_required_when_system_ffmpeg_missing(
        self,
        find_spec_mock,
        which_mock,
    ) -> None:
        which_mock.return_value = None

        def fake_find_spec(module_name: str) -> object | None:
            if module_name in {"torch", "whisper"}:
                return object()
            return None

        find_spec_mock.side_effect = fake_find_spec

        missing_packages = collect_missing_transcription_packages()

        self.assertEqual(["imageio-ffmpeg"], missing_packages)

    @patch("whisper_sift.runtime.dependencies.probe_nvidia_hardware")
    def test_missing_torch_prefers_cuda_wheel_when_nvidia_is_detected(
        self,
        probe_nvidia_hardware_mock,
    ) -> None:
        probe_nvidia_hardware_mock.return_value = NvidiaHardwareStatus(
            available=True,
            executable="nvidia-smi",
            gpu_names=("NVIDIA RTX",),
        )

        commands = build_dependency_install_commands(["torch", "openai-whisper"])

        self.assertEqual(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                TORCH_CUDA_INDEX_URL,
            ],
            commands[0],
        )
        self.assertEqual(
            [sys.executable, "-m", "pip", "install", "openai-whisper"],
            commands[1],
        )

    @patch("whisper_sift.runtime.dependencies.probe_nvidia_hardware")
    def test_missing_torch_falls_back_to_cpu_wheel_without_nvidia(
        self,
        probe_nvidia_hardware_mock,
    ) -> None:
        probe_nvidia_hardware_mock.return_value = NvidiaHardwareStatus(
            available=False,
            executable=None,
        )

        command = build_torch_install_command()

        self.assertEqual(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "torch",
                "--index-url",
                TORCH_CPU_INDEX_URL,
            ],
            command,
        )


if __name__ == "__main__":
    unittest.main()
