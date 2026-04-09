from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.runtime.hardware import probe_nvidia_hardware


class HardwareProbeTests(unittest.TestCase):
    @patch("whisper_sift.runtime.hardware.shutil.which")
    def test_probe_reports_not_found_when_nvidia_smi_missing(
        self,
        which_mock,
    ) -> None:
        which_mock.return_value = None

        status = probe_nvidia_hardware()

        self.assertFalse(status.available)
        self.assertIsNone(status.executable)
        self.assertEqual((), status.gpu_names)
        self.assertIsNone(status.error)

    @patch("whisper_sift.runtime.hardware.subprocess.run")
    @patch("whisper_sift.runtime.hardware.shutil.which")
    def test_probe_collects_gpu_names_from_nvidia_smi(
        self,
        which_mock,
        run_mock,
    ) -> None:
        which_mock.return_value = "nvidia-smi"
        run_mock.return_value = Mock(
            stdout="NVIDIA RTX 4070 Laptop GPU\nNVIDIA RTX 5000\n"
        )

        status = probe_nvidia_hardware()

        self.assertTrue(status.available)
        self.assertEqual("nvidia-smi", status.executable)
        self.assertEqual(
            ("NVIDIA RTX 4070 Laptop GPU", "NVIDIA RTX 5000"),
            status.gpu_names,
        )
        self.assertIsNone(status.error)

    @patch("whisper_sift.runtime.hardware.subprocess.run")
    @patch("whisper_sift.runtime.hardware.shutil.which")
    def test_probe_captures_query_error(
        self,
        which_mock,
        run_mock,
    ) -> None:
        which_mock.return_value = "nvidia-smi"
        run_mock.side_effect = subprocess.CalledProcessError(
            returncode=1,
            cmd=["nvidia-smi"],
            stderr="query failed",
        )

        status = probe_nvidia_hardware()

        self.assertFalse(status.available)
        self.assertEqual("nvidia-smi", status.executable)
        self.assertEqual((), status.gpu_names)
        self.assertIsNotNone(status.error)


if __name__ == "__main__":
    unittest.main()
