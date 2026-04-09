from whisper_sift.cli_handlers.doctor import handle_doctor
from whisper_sift.cli_handlers.evaluate import handle_evaluate
from whisper_sift.cli_handlers.extract_questions import handle_extract_questions
from whisper_sift.cli_handlers.pipeline import handle_pipeline
from whisper_sift.cli_handlers.transcribe import handle_transcribe

__all__ = [
    "handle_doctor",
    "handle_evaluate",
    "handle_extract_questions",
    "handle_pipeline",
    "handle_transcribe",
]
