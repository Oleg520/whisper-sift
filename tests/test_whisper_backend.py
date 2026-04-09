from __future__ import annotations

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.config import FAKE_TRANSCRIPTION_FILE_ENV, FAKE_TRANSCRIPTION_TEXT_ENV
from whisper_sift.infrastructure.whisper_backend import (
    FixtureWhisperBackend,
    load_whisper_backend,
    load_fake_transcription_text,
    resolve_backend_device,
)


class WhisperBackendTests(unittest.TestCase):
    def test_load_fake_transcription_text_reads_fixture_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture_path = Path(temp_dir) / "fixture.txt"
            fixture_path.write_text("hello", encoding="utf-8")

            with patch.dict(os.environ, {FAKE_TRANSCRIPTION_FILE_ENV: str(fixture_path)}, clear=False):
                self.assertEqual("hello", load_fake_transcription_text())

    def test_load_whisper_backend_uses_fixture_backend_when_env_is_set(self) -> None:
        with patch.dict(
            os.environ,
            {FAKE_TRANSCRIPTION_TEXT_ENV: "Interviewer: Hello there"},
            clear=False,
        ):
            backend = load_whisper_backend("small", "cuda")

        self.assertIsInstance(backend, FixtureWhisperBackend)
        self.assertEqual("cpu", backend.resolved_device)
        self.assertFalse(backend.use_fp16)
        self.assertFalse(backend.requires_media_runtime)

    def test_resolve_backend_device_falls_back_to_cpu_when_cuda_is_unavailable(self) -> None:
        fake_torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(is_available=lambda: False),
            backends=types.SimpleNamespace(
                mps=types.SimpleNamespace(is_available=lambda: False)
            ),
        )

        self.assertEqual(
            "cpu",
            resolve_backend_device("cuda", torch_module=fake_torch),
        )


if __name__ == "__main__":
    unittest.main()
