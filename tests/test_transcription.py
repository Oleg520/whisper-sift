from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from whisper_sift.config import TranscriptionOptions
from whisper_sift.domain.transcription import TranscriptionDocument
from whisper_sift.services.transcription import transcribe_files, transcribe_sources


class TranscriptionTests(unittest.TestCase):
    @patch("whisper_sift.services.transcription.ensure_ffmpeg_on_path")
    @patch("whisper_sift.services.transcription.load_whisper_backend")
    def test_missing_input_fails_before_runtime_setup(
        self,
        load_backend_mock,
        ensure_ffmpeg_mock,
    ) -> None:
        missing_file = PROJECT_ROOT / "missing_audio.mkv"
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir) / "results"

            with self.assertRaises(FileNotFoundError):
                transcribe_files(
                    TranscriptionOptions(
                        files=[missing_file],
                        output_dir=output_dir,
                    )
                )

        load_backend_mock.assert_not_called()
        ensure_ffmpeg_mock.assert_not_called()
        self.assertFalse(output_dir.exists())

    @patch("whisper_sift.services.transcription.write_whisper_outputs")
    @patch("whisper_sift.services.transcription.ensure_ffmpeg_on_path")
    @patch("whisper_sift.services.transcription.load_whisper_backend")
    def test_transcribe_files_uses_backend_adapter(
        self,
        load_backend_mock,
        ensure_ffmpeg_mock,
        write_outputs_mock,
    ) -> None:
        backend = Mock()
        backend.model_name = "small"
        backend.resolved_device = "cpu"
        backend.use_fp16 = False
        backend.requires_media_runtime = True
        backend.transcribe_file.return_value = TranscriptionDocument(text="hello")
        load_backend_mock.return_value = backend
        ensure_ffmpeg_mock.return_value = Path("/usr/bin/ffmpeg")

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source = workspace / "interview.mkv"
            source.write_bytes(b"fake media")
            output_dir = workspace / "results"
            write_outputs_mock.return_value = [output_dir / "interview.txt"]

            result = transcribe_files(
                TranscriptionOptions(
                    files=[source],
                    output_dir=output_dir,
                    formats=("txt",),
                )
            )

        load_backend_mock.assert_called_once_with("small", "auto", reporter=None)
        ensure_ffmpeg_mock.assert_called_once_with()
        backend.transcribe_file.assert_called_once()
        write_outputs_mock.assert_called_once()
        self.assertIsInstance(
            write_outputs_mock.call_args.kwargs["result"],
            TranscriptionDocument,
        )
        self.assertEqual([output_dir / "interview.txt"], result)

    @patch("whisper_sift.services.transcription.write_whisper_outputs")
    @patch("whisper_sift.services.transcription.ensure_ffmpeg_on_path")
    @patch("whisper_sift.services.transcription.load_whisper_backend")
    def test_transcribe_sources_returns_structured_artifacts(
        self,
        load_backend_mock,
        ensure_ffmpeg_mock,
        write_outputs_mock,
    ) -> None:
        backend = Mock()
        backend.model_name = "small"
        backend.resolved_device = "cpu"
        backend.use_fp16 = False
        backend.requires_media_runtime = True
        backend.transcribe_file.return_value = TranscriptionDocument(
            text="hello",
            language="ru",
        )
        load_backend_mock.return_value = backend
        ensure_ffmpeg_mock.return_value = Path("/usr/bin/ffmpeg")

        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            source = workspace / "interview.mkv"
            source.write_bytes(b"fake media")
            output_dir = workspace / "results"
            write_outputs_mock.return_value = [
                output_dir / "interview.txt",
                output_dir / "interview.srt",
            ]

            result = transcribe_sources(
                TranscriptionOptions(
                    files=[source],
                    output_dir=output_dir,
                    formats=("txt", "srt"),
                )
            )

        self.assertEqual(1, len(result.artifacts))
        artifact = result.artifacts[0]
        self.assertEqual(source.resolve(), artifact.source_path)
        self.assertEqual("small", artifact.model_name)
        self.assertEqual(("txt", "srt"), tuple(output.output_format for output in artifact.outputs))
        self.assertEqual(
            (output_dir / "interview.txt", output_dir / "interview.srt"),
            artifact.generated_files,
        )


if __name__ == "__main__":
    unittest.main()
