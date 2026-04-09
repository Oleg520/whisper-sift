from __future__ import annotations

import re
from collections import Counter

from whisper_sift.domain.extraction_text import normalize_whitespace, tokenize_question
from whisper_sift.domain.transcript import SpeakerTurn, TranscriptSlice

GENERIC_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]{1,40})\]|(?P<plain>[^:\n]{1,40}?))"
    r"(?:\s*:\s*|\s+[-–—]\s+)(?P<body>.*)$"
)
KNOWN_INTERVIEWER_LABELS = (
    "interviewer",
    "интервьюер",
    "recruiter",
    "рекрутер",
    "hr",
    "question",
    "вопрос",
    "q",
    "собеседующий",
)
KNOWN_CANDIDATE_LABELS = (
    "candidate",
    "кандидат",
    "соискатель",
    "interviewee",
    "answer",
    "ответ",
    "a",
)
KNOWN_SPEAKER_LABEL_RE = re.compile(
    r"^(?:speaker|spk|спикер)[ _-]?\d+$",
    re.IGNORECASE,
)
MAX_ROLE_LABEL_TOKENS = 4


def label_transcript_slices(
    slices: list[TranscriptSlice],
    *,
    explicit_labels: set[str],
) -> tuple[list[TranscriptSlice], list[SpeakerTurn]]:
    candidate_labels: list[str] = []
    parsed_slices: list[tuple[TranscriptSlice, str | None, str | None, str | None]] = []

    for transcript_slice in slices:
        speaker_data = split_speaker_labeled_text(transcript_slice.text)
        if speaker_data is None:
            parsed_slices.append((transcript_slice, None, None, None))
            continue

        raw_label, normalized_label, body = speaker_data
        if normalized_label:
            candidate_labels.append(normalized_label)
        parsed_slices.append((transcript_slice, raw_label.strip(), normalized_label, body))

    label_counts = Counter(candidate_labels)
    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= 2
        or label in explicit_labels
        or looks_like_known_speaker_label(label)
    }
    if not valid_labels:
        return slices, []

    labeled_slices: list[TranscriptSlice] = []
    speaker_turns: list[SpeakerTurn] = []
    for transcript_slice, raw_label, normalized_label, body in parsed_slices:
        if raw_label and normalized_label in valid_labels and body:
            labeled_slice = TranscriptSlice(
                text=body,
                speaker_label=raw_label,
                normalized_speaker_label=normalized_label,
                start_time=transcript_slice.start_time,
                end_time=transcript_slice.end_time,
            )
            labeled_slices.append(labeled_slice)
            speaker_turns.append(
                SpeakerTurn(
                    label=raw_label,
                    normalized_label=normalized_label,
                    text=body,
                )
            )
            continue

        labeled_slices.append(transcript_slice)

    return labeled_slices, speaker_turns


def split_speaker_labeled_text(
    value: str,
) -> tuple[str, str, str] | None:
    match = GENERIC_SPEAKER_LINE_RE.match(value)
    if not match:
        return None

    raw_label = match.group("bracket") or match.group("plain") or ""
    normalized_label = normalize_speaker_label(raw_label)
    body = normalize_whitespace(match.group("body") or "")
    if not normalized_label or not body:
        return None
    return raw_label.strip(), normalized_label, body


def extract_speaker_turns(
    text: str,
    *,
    interviewer_labels: tuple[str, ...],
) -> list[SpeakerTurn]:
    explicit_labels = normalize_explicit_speaker_labels(interviewer_labels)
    lines = text.splitlines()

    candidate_labels: list[str] = []
    for line in lines:
        match = GENERIC_SPEAKER_LINE_RE.match(line)
        if not match:
            continue
        raw_label = match.group("bracket") or match.group("plain") or ""
        normalized_label = normalize_speaker_label(raw_label)
        if normalized_label:
            candidate_labels.append(normalized_label)

    label_counts = Counter(candidate_labels)
    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= 2
        or label in explicit_labels
        or looks_like_known_speaker_label(label)
    }
    if not valid_labels:
        return []

    turns: list[SpeakerTurn] = []
    for raw_line in lines:
        line = raw_line.strip()
        match = GENERIC_SPEAKER_LINE_RE.match(raw_line)
        if match:
            raw_label = match.group("bracket") or match.group("plain") or ""
            normalized_label = normalize_speaker_label(raw_label)
            if normalized_label in valid_labels:
                body = (match.group("body") or "").strip()
                turns.append(
                    SpeakerTurn(
                        label=raw_label.strip(),
                        normalized_label=normalized_label,
                        text=body,
                    )
                )
                continue

        if turns and line:
            previous_turn = turns[-1]
            turns[-1] = SpeakerTurn(
                label=previous_turn.label,
                normalized_label=previous_turn.normalized_label,
                text=f"{previous_turn.text}\n{line}".strip(),
            )

    return [turn for turn in turns if turn.text]


def resolve_interviewer_labels(
    turns: list[SpeakerTurn],
    *,
    interviewer_labels: tuple[str, ...],
) -> set[str]:
    if not turns:
        return set()

    available_labels = {turn.normalized_label for turn in turns}
    explicit_labels = normalize_explicit_speaker_labels(interviewer_labels)
    if explicit_labels:
        return available_labels & explicit_labels

    detected_labels = {
        label
        for label in available_labels
        if label_matches_any_keywords(
            label,
            KNOWN_INTERVIEWER_LABELS,
            require_role_like=True,
        )
    }
    if detected_labels:
        return detected_labels

    candidate_labels = {
        label
        for label in available_labels
        if label_matches_any_keywords(
            label,
            KNOWN_CANDIDATE_LABELS,
            require_role_like=True,
        )
    }
    remaining_labels = available_labels - candidate_labels
    if len(available_labels) == 2 and len(candidate_labels) == 1 and len(remaining_labels) == 1:
        return remaining_labels

    return set()


def normalize_speaker_label(value: str) -> str:
    normalized = value.strip().strip("[]")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalize_whitespace(normalized).lower()
    return normalized


def normalize_explicit_speaker_labels(labels: tuple[str, ...]) -> set[str]:
    return {
        normalized
        for label in labels
        if (normalized := normalize_speaker_label(label))
    }


def label_matches_any_keywords(
    label: str,
    keywords: tuple[str, ...],
    *,
    require_role_like: bool = False,
) -> bool:
    return any(
        label_matches_keyword(
            label,
            keyword,
            require_role_like=require_role_like,
        )
        for keyword in keywords
    )


def label_matches_keyword(
    label: str,
    keyword: str,
    *,
    require_role_like: bool = False,
) -> bool:
    label_tokens = tokenize_question(label)
    keyword_tokens = tokenize_question(keyword)
    if not label_tokens or not keyword_tokens:
        return False
    if require_role_like and not label_is_role_like(label_tokens, keyword_tokens):
        return False

    if len(keyword_tokens) == 1:
        return keyword_tokens[0] in label_tokens

    window_size = len(keyword_tokens)
    for index in range(len(label_tokens) - window_size + 1):
        if label_tokens[index : index + window_size] == keyword_tokens:
            return True
    return False


def label_is_role_like(
    label_tokens: list[str],
    keyword_tokens: list[str],
) -> bool:
    max_tokens = max(len(keyword_tokens) + 2, 3)
    max_tokens = min(max_tokens, MAX_ROLE_LABEL_TOKENS)
    return len(label_tokens) <= max_tokens


def looks_like_known_speaker_label(label: str) -> bool:
    if not label:
        return False
    if KNOWN_SPEAKER_LABEL_RE.match(label):
        return True
    if label_matches_any_keywords(
        label,
        KNOWN_INTERVIEWER_LABELS,
        require_role_like=True,
    ):
        return True
    if label_matches_any_keywords(
        label,
        KNOWN_CANDIDATE_LABELS,
        require_role_like=True,
    ):
        return True
    return False


def ordered_unique(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered
