from __future__ import annotations

import re
from pathlib import Path

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_INTERVIEWER_LABELS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
)
import whisper_sift.domain.question_classifier as _question_classifier
import whisper_sift.domain.question_normalizer as _question_normalizer
from whisper_sift.domain.extraction_text import (
    deduplicate_preserving_order as _deduplicate_preserving_order,
    normalize_whitespace,
)
from whisper_sift.domain.questions import QuestionCandidate, QuestionExtractionResult
from whisper_sift.domain.srt_segmenter import extract_srt_slices as _extract_srt_slices
from whisper_sift.domain.speaker_resolution import (
    extract_speaker_turns as _extract_speaker_turns,
    label_transcript_slices as _label_transcript_slices,
    normalize_explicit_speaker_labels as _normalize_explicit_speaker_labels,
    ordered_unique as _ordered_unique,
    resolve_interviewer_labels as _resolve_interviewer_labels,
    split_speaker_labeled_text as _split_speaker_labeled_text,
)
from whisper_sift.domain.transcript import (
    TranscriptArtifact,
    TranscriptSlice,
    TranscriptView,
)


QUESTION_SPLIT_RE = re.compile(r"(?<=[?.!])\s+|\n+")
CANDIDATE_QUESTION_INVITE_RE = re.compile(
    r"(?:"
    r"есть\s+ли\s+у\s+вас\s+вопросы|"
    r"у\s+вас\s+есть\s+вопросы|"
    r"у\s+вас\s+какие[- ]?то\s+вопросы|"
    r"может(?:,\s*|\s+)у\s+вас\s+какие[- ]?то\s+вопросы|"
    r"может(?:,\s*|\s+)есть\s+какие[- ]?то\s+вопросы"
    r")",
    re.IGNORECASE,
)


def extract_questions(
    text: str,
    *,
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS,
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH,
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH,
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS,
    source_name: str | None = None,
) -> QuestionExtractionResult:
    artifact = TranscriptArtifact(text=text, source_name=source_name)
    transcript = _build_transcript_view(
        artifact=artifact,
        interviewer_labels=interviewer_labels,
    )

    questions: list[QuestionCandidate] = []
    previous_candidate: QuestionCandidate | None = None
    for transcript_slice in transcript.candidate_slices:
        extracted_questions = _extract_questions_from_slice(
            transcript_slice,
            min_length=min_length,
            max_length=max_length,
            previous_candidate=previous_candidate,
        )
        questions.extend(extracted_questions)
        if extracted_questions:
            previous_candidate = extracted_questions[-1]

    if deduplicate:
        questions = _deduplicate_preserving_order(questions)

    return QuestionExtractionResult(
        transcript=transcript,
        questions=tuple(questions),
        deduplicated=deduplicate,
        min_length=min_length,
        max_length=max_length,
    )


def extract_question_candidates(
    text: str,
    *,
    deduplicate: bool = DEFAULT_DEDUPLICATE_QUESTIONS,
    min_length: int = DEFAULT_MIN_QUESTION_LENGTH,
    max_length: int = DEFAULT_MAX_QUESTION_LENGTH,
    interviewer_labels: tuple[str, ...] = DEFAULT_INTERVIEWER_LABELS,
    source_name: str | None = None,
) -> list[str]:
    result = extract_questions(
        text,
        deduplicate=deduplicate,
        min_length=min_length,
        max_length=max_length,
        interviewer_labels=interviewer_labels,
        source_name=source_name,
    )
    return list(result.question_texts)


def _build_transcript_view(
    *,
    artifact: TranscriptArtifact,
    interviewer_labels: tuple[str, ...],
) -> TranscriptView:
    if _is_srt_source(artifact.source_name):
        return _build_srt_transcript_view(
            artifact=artifact,
            interviewer_labels=interviewer_labels,
        )

    explicit_interviewer_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    speaker_turns = _extract_speaker_turns(
        artifact.text,
        interviewer_labels=interviewer_labels,
    )
    speaker_filter = _resolve_interviewer_labels(
        speaker_turns,
        interviewer_labels=interviewer_labels,
    )

    if explicit_interviewer_labels:
        if not speaker_turns:
            raise RuntimeError(
                "Explicit interviewer labels were provided, but no speaker-labeled "
                "transcript structure was detected."
            )
        if not speaker_filter:
            available_labels = ", ".join(_ordered_unique(turn.label for turn in speaker_turns))
            raise RuntimeError(
                "None of the provided interviewer labels were found in the transcript. "
                f"Available labels: {available_labels}"
            )

    if speaker_turns and speaker_filter:
        candidate_slices = tuple(
            TranscriptSlice(
                text=turn.text,
                speaker_label=turn.label,
                normalized_speaker_label=turn.normalized_label,
            )
            for turn in speaker_turns
            if turn.normalized_label in speaker_filter
        )
    else:
        candidate_slices = (TranscriptSlice(text=artifact.text),)

    return TranscriptView(
        artifact=artifact,
        speaker_turns=tuple(speaker_turns),
        candidate_slices=candidate_slices,
        selected_interviewer_labels=tuple(sorted(speaker_filter)),
        available_speaker_labels=tuple(_ordered_unique(turn.label for turn in speaker_turns)),
    )


def _build_srt_transcript_view(
    *,
    artifact: TranscriptArtifact,
    interviewer_labels: tuple[str, ...],
) -> TranscriptView:
    explicit_interviewer_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    raw_slices = _extract_srt_slices(
        artifact.text,
        normalize_whitespace=normalize_whitespace,
        cleanup_question=_question_classifier.cleanup_question,
        looks_like_question_without_mark=_question_classifier.looks_like_question_without_mark,
        looks_like_answer=_question_classifier.looks_like_answer,
        split_speaker_labeled_text=_split_speaker_labeled_text,
    )
    labeled_slices, speaker_turns = _label_transcript_slices(
        raw_slices,
        explicit_labels=explicit_interviewer_labels,
    )
    speaker_filter = _resolve_interviewer_labels(
        speaker_turns,
        interviewer_labels=interviewer_labels,
    )

    if explicit_interviewer_labels:
        if not speaker_turns:
            raise RuntimeError(
                "Explicit interviewer labels were provided, but no speaker-labeled "
                "transcript structure was detected."
            )
        if not speaker_filter:
            available_labels = ", ".join(_ordered_unique(turn.label for turn in speaker_turns))
            raise RuntimeError(
                "None of the provided interviewer labels were found in the transcript. "
                f"Available labels: {available_labels}"
            )

    if speaker_turns and speaker_filter:
        candidate_slices = tuple(
            transcript_slice
            for transcript_slice in labeled_slices
            if transcript_slice.normalized_speaker_label in speaker_filter
        )
    elif raw_slices:
        candidate_slices = tuple(labeled_slices)
    else:
        candidate_slices = (TranscriptSlice(text=artifact.text),)

    return TranscriptView(
        artifact=artifact,
        speaker_turns=tuple(speaker_turns),
        candidate_slices=candidate_slices,
        selected_interviewer_labels=tuple(sorted(speaker_filter)),
        available_speaker_labels=tuple(_ordered_unique(turn.label for turn in speaker_turns)),
    )


def _extract_questions_from_slice(
    transcript_slice: TranscriptSlice,
    *,
    min_length: int,
    max_length: int,
    previous_candidate: QuestionCandidate | None,
) -> list[QuestionCandidate]:
    prepared_text = _trim_after_candidate_question_invite(
        normalize_whitespace(transcript_slice.text)
    )
    chunks = QUESTION_SPLIT_RE.split(prepared_text)

    questions: list[QuestionCandidate] = []
    local_previous_candidate = previous_candidate
    for chunk in chunks:
        parts = chunk.split("?") if "?" in chunk else [chunk]

        for part in parts:
            cleaned = _question_classifier.cleanup_question(part)
            if not cleaned:
                continue

            is_explicit_question = "?" in chunk
            if (
                not is_explicit_question
                and not _question_classifier.looks_like_question_without_mark(cleaned)
            ):
                continue

            question_text = cleaned if cleaned.endswith("?") else f"{cleaned}?"
            if len(question_text) < min_length or len(question_text) > max_length:
                continue

            classification = _question_classifier.classify_question_candidate(
                question_text=question_text,
                explicit_question=is_explicit_question,
                previous_candidate=local_previous_candidate,
            )
            if not classification.accepted:
                continue

            normalized_question_text = _question_normalizer.normalize_question_for_output(
                question_text,
                previous_candidate=local_previous_candidate,
            )
            candidate = QuestionCandidate(
                text=normalized_question_text,
                source_text=cleaned,
                explicit_question=is_explicit_question,
                speaker_label=transcript_slice.speaker_label,
                normalized_speaker_label=transcript_slice.normalized_speaker_label,
                start_time=transcript_slice.start_time,
                end_time=transcript_slice.end_time,
            )
            questions.append(candidate)
            local_previous_candidate = candidate

    return questions


def _trim_after_candidate_question_invite(value: str) -> str:
    match = CANDIDATE_QUESTION_INVITE_RE.search(value)
    if match is None:
        return value
    return value[: match.start()].rstrip()


def _is_srt_source(source_name: str | None) -> bool:
    if not source_name:
        return False
    return Path(source_name).suffix.lower() == ".srt"
