from __future__ import annotations

import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import whisper_sift.paths as paths


class PathTests(unittest.TestCase):
    def test_runtime_root_can_be_overridden(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            override = Path(temp_dir) / "custom-runtime"

            with patch.dict(os.environ, {"WHISPER_SIFT_RUNTIME_DIR": str(override)}):
                reloaded = importlib.reload(paths)
                self.assertEqual(override.resolve(), reloaded.RUNTIME_ROOT)
                self.assertEqual(reloaded.RUNTIME_ROOT / "tools", reloaded.TOOLS_DIR)

        importlib.reload(paths)

    def test_project_root_is_detected_in_repo_checkout(self) -> None:
        self.assertEqual(PROJECT_ROOT, paths.PROJECT_ROOT)


if __name__ == "__main__":
    unittest.main()
