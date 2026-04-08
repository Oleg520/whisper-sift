from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass


WHITESPACE_RE = re.compile(r"[ \t]+")
QUESTION_SPLIT_RE = re.compile(r"(?<=[?.!])\s+|\n+")
GENERIC_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]{1,40})\]|(?P<plain>[^:\-\n]{1,40}))\s*[:\-]\s*(?P<body>.*)$"
)
QUESTION_LIKE_RE = re.compile(
    r"^(?:"
    r"что|как|почему|зачем|когда|где|кто|сколько|какой|какая|какие|какое|каким|какую|"
    r"чем|можете|можешь|можно|есть ли|был ли|были ли|"
    r"расскажите|расскажи|подскажите|объясните|верно ли|правильно ли|"
    r"what|how|why|when|where|who|which|can you|could you|would you|"
    r"do you|did you|have you|is there|are there|tell me"
    r")\b",
    re.IGNORECASE,
)
ANSWER_LIKE_RE = re.compile(
    r"^(?:"
    r"да|нет|ага|ну|смотрите|слушайте|хорошо|конечно|скорее|получается|"
    r"я|мы|это|было|есть|наверное|в целом|actually|well|yes|no|i|we|it"
    r")\b",
    re.IGNORECASE,
)
CANONICAL_WHITESPACE_RE = re.compile(r"\s+")
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


@dataclass(slots=True)
class SpeakerTurn:
    label: str
    normalized_label: str
    text: str


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = WHITESPACE_RE.sub(" ", value)
    value = re.sub(r" ?\n ?", "\n", value)
    return value.strip()


def extract_question_candidates(
    text: str,
    *,
    deduplicate: bool = True,
    min_length: int = 10,
    max_length: int = 240,
    interviewer_labels: tuple[str, ...] = (),
) -> list[str]:
    questions: list[str] = []
    speaker_turns = _extract_speaker_turns(text, interviewer_labels=interviewer_labels)
    speaker_filter = _resolve_interviewer_labels(
        speaker_turns,
        interviewer_labels=interviewer_labels,
    )

    if speaker_turns and speaker_filter:
        candidate_texts = [
            turn.text
            for turn in speaker_turns
            if turn.normalized_label in speaker_filter
        ]
    else:
        candidate_texts = [text]

    for candidate_text in candidate_texts:
        questions.extend(
            _extract_questions_from_text(
                candidate_text,
                min_length=min_length,
                max_length=max_length,
            )
        )

    if deduplicate:
        return _deduplicate_preserving_order(questions)
    return questions


def _extract_questions_from_text(
    text: str,
    *,
    min_length: int,
    max_length: int,
) -> list[str]:
    prepared_text = normalize_whitespace(text)
    chunks = QUESTION_SPLIT_RE.split(prepared_text)

    questions: list[str] = []
    for chunk in chunks:
        parts = chunk.split("?") if "?" in chunk else [chunk]

        for part in parts:
            cleaned = _cleanup_question(part)
            if not cleaned:
                continue

            is_explicit_question = "?" in chunk
            if not is_explicit_question and not _looks_like_question_without_mark(cleaned):
                continue

            question = cleaned if cleaned.endswith("?") else f"{cleaned}?"
            if len(question) < min_length or len(question) > max_length:
                continue
            if _looks_like_noise(question):
                continue
            questions.append(question)

    return questions


def _cleanup_question(value: str) -> str:
    cleaned = normalize_whitespace(value)
    cleaned = cleaned.strip(" .,!;:-")
    cleaned = cleaned.replace(" ?", "?")
    return cleaned


def _looks_like_noise(value: str) -> bool:
    tokens = re.findall(r"[A-Za-zА-Яа-яЁё0-9-]+", value.lower())
    if not tokens:
        return True

    unique_ratio = len(set(tokens)) / len(tokens)
    if len(tokens) >= 8 and unique_ratio < 0.35:
        return True

    letters = re.findall(r"[A-Za-zА-Яа-яЁё]", value)
    return len(letters) < 5


def _looks_like_question_without_mark(value: str) -> bool:
    lowered = normalize_whitespace(value).lower()
    if not lowered:
        return False
    if ANSWER_LIKE_RE.match(lowered):
        return False
    return QUESTION_LIKE_RE.match(lowered) is not None


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        canonical = _canonicalize_question(value)
        if canonical in seen:
            continue
        seen.add(canonical)
        result.append(value)

    return result


def _canonicalize_question(value: str) -> str:
    normalized = CANONICAL_WHITESPACE_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")


def _extract_speaker_turns(
    text: str,
    *,
    interviewer_labels: tuple[str, ...],
) -> list[SpeakerTurn]:
    explicit_labels = {
        normalized
        for label in interviewer_labels
        if (normalized := _normalize_speaker_label(label))
    }
    lines = text.splitlines()

    candidate_labels: list[str] = []
    for line in lines:
        match = GENERIC_SPEAKER_LINE_RE.match(line)
        if not match:
            continue
        raw_label = match.group("bracket") or match.group("plain") or ""
        normalized_label = _normalize_speaker_label(raw_label)
        if normalized_label:
            candidate_labels.append(normalized_label)

    label_counts = Counter(candidate_labels)
    valid_labels = {
        label
        for label, count in label_counts.items()
        if count >= 2
        or label in explicit_labels
        or _looks_like_known_speaker_label(label)
    }
    if not valid_labels:
        return []

    turns: list[SpeakerTurn] = []
    for raw_line in lines:
        line = raw_line.strip()
        match = GENERIC_SPEAKER_LINE_RE.match(raw_line)
        if match:
            raw_label = match.group("bracket") or match.group("plain") or ""
            normalized_label = _normalize_speaker_label(raw_label)
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
            turns[-1].text = f"{turns[-1].text}\n{line}".strip()

    return [turn for turn in turns if turn.text]


def _resolve_interviewer_labels(
    turns: list[SpeakerTurn],
    *,
    interviewer_labels: tuple[str, ...],
) -> set[str]:
    if not turns:
        return set()

    available_labels = {turn.normalized_label for turn in turns}
    explicit_labels = {
        normalized
        for label in interviewer_labels
        if (normalized := _normalize_speaker_label(label))
    }
    if explicit_labels:
        return available_labels & explicit_labels

    detected_labels = {
        label
        for label in available_labels
        if any(keyword in label for keyword in KNOWN_INTERVIEWER_LABELS)
    }
    if detected_labels:
        return detected_labels

    candidate_labels = {
        label
        for label in available_labels
        if any(keyword in label for keyword in KNOWN_CANDIDATE_LABELS)
    }
    remaining_labels = available_labels - candidate_labels
    if len(available_labels) == 2 and len(candidate_labels) == 1 and len(remaining_labels) == 1:
        return remaining_labels

    return set()


def _normalize_speaker_label(value: str) -> str:
    normalized = value.strip().strip("[]")
    normalized = normalized.replace("_", " ").replace("-", " ")
    normalized = normalize_whitespace(normalized).lower()
    return normalized


def _looks_like_known_speaker_label(label: str) -> bool:
    if not label:
        return False
    if KNOWN_SPEAKER_LABEL_RE.match(label):
        return True
    if any(keyword in label for keyword in KNOWN_INTERVIEWER_LABELS):
        return True
    if any(keyword in label for keyword in KNOWN_CANDIDATE_LABELS):
        return True
    return False
