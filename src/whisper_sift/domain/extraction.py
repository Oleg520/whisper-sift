from __future__ import annotations

import re
from collections import Counter
from difflib import SequenceMatcher

from whisper_sift.config import (
    DEFAULT_DEDUPLICATE_QUESTIONS,
    DEFAULT_INTERVIEWER_LABELS,
    DEFAULT_MAX_QUESTION_LENGTH,
    DEFAULT_MIN_QUESTION_LENGTH,
)
from whisper_sift.domain.questions import QuestionCandidate, QuestionExtractionResult
from whisper_sift.domain.transcript import (
    SpeakerTurn,
    TranscriptArtifact,
    TranscriptSlice,
    TranscriptView,
)


WHITESPACE_RE = re.compile(r"[ \t]+")
QUESTION_SPLIT_RE = re.compile(r"(?<=[?.!])\s+|\n+")
GENERIC_SPEAKER_LINE_RE = re.compile(
    r"^\s*(?:\[(?P<bracket>[^\]]{1,40})\]|(?P<plain>[^:\-\n]{1,40}))\s*[:\-]\s*(?P<body>.*)$"
)
QUESTION_LIKE_RE = re.compile(
    r"^(?:"
    r"что(?!-)|как(?!-)|почему|зачем|когда(?!-)|где(?!-)|кто(?!-)|сколько|"
    r"какой(?!-)|какая(?!-)|какие(?!-)|какое(?!-)|каким(?!-)|какую(?!-)|какого(?!-)|какому(?!-)|"
    r"чем(?!-)|можете|можешь|можно|есть ли|был ли|были ли|"
    r"расскажите|расскажи|подскажите|объясните|верно ли|правильно ли|"
    r"я правильно понимаю|я верно понимаю|если я правильно понимаю|"
    r"правильно понимаю|верно понимаю|"
    r"what|how|why|when|where|who|which|can you|could you|would you|"
    r"do you|did you|have you|is there|are there|tell me"
    r")\b",
    re.IGNORECASE,
)
ANSWER_LIKE_RE = re.compile(
    r"^(?:"
    r"да|нет|ага|ну|смотрите|слушайте|хорошо|конечно|скорее|получается|"
    r"наверное|в целом|на самом деле|если честно|честно говоря|не помню|"
    r"я|мы|мне|нам|у нас|в нашей команде|в компании|на проекте|на последнем проекте|"
    r"на прошлом проекте|обычно я|обычно мы|это|было|есть|"
    r"actually|well|yes|no|i|we|our team|in our team|on the last project"
    r")\b",
    re.IGNORECASE,
)
CANONICAL_WHITESPACE_RE = re.compile(r"\s+")
QUESTION_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
LEADING_FILLER_RE = re.compile(
    r"^(?:(?:а|ну|вот|так|и|слушай|смотри)\s*[,:\-]?\s*)+",
    re.IGNORECASE,
)
FILLER_TOKEN_RE = re.compile(
    r"^(?:а|ну|вот|так|и|то|есть|да|наверное|если|честно|вообще|как|бы|смотри|слушай)$",
    re.IGNORECASE,
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


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = WHITESPACE_RE.sub(" ", value)
    value = re.sub(r" ?\n ?", "\n", value)
    return value.strip()


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
    for transcript_slice in transcript.candidate_slices:
        questions.extend(
            _extract_questions_from_slice(
                transcript_slice,
                min_length=min_length,
                max_length=max_length,
            )
        )

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
) -> list[str]:
    result = extract_questions(
        text,
        deduplicate=deduplicate,
        min_length=min_length,
        max_length=max_length,
        interviewer_labels=interviewer_labels,
    )
    return list(result.question_texts)


def _build_transcript_view(
    *,
    artifact: TranscriptArtifact,
    interviewer_labels: tuple[str, ...],
) -> TranscriptView:
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


def _extract_questions_from_slice(
    transcript_slice: TranscriptSlice,
    *,
    min_length: int,
    max_length: int,
) -> list[QuestionCandidate]:
    prepared_text = normalize_whitespace(transcript_slice.text)
    chunks = QUESTION_SPLIT_RE.split(prepared_text)

    questions: list[QuestionCandidate] = []
    for chunk in chunks:
        parts = chunk.split("?") if "?" in chunk else [chunk]

        for part in parts:
            cleaned = _cleanup_question(part)
            if not cleaned:
                continue

            is_explicit_question = "?" in chunk
            if not is_explicit_question and not _looks_like_question_without_mark(cleaned):
                continue

            question_text = cleaned if cleaned.endswith("?") else f"{cleaned}?"
            if _looks_like_answer(question_text):
                continue
            if _looks_like_filler_fragment(question_text):
                continue
            if len(question_text) < min_length or len(question_text) > max_length:
                continue
            if _looks_like_noise(question_text):
                continue

            questions.append(
                QuestionCandidate(
                    text=question_text,
                    source_text=cleaned,
                    explicit_question=is_explicit_question,
                    speaker_label=transcript_slice.speaker_label,
                    normalized_speaker_label=transcript_slice.normalized_speaker_label,
                )
            )

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
    lowered = _normalize_intent_text(value)
    if not lowered:
        return False
    if QUESTION_LIKE_RE.match(lowered):
        return True
    if ANSWER_LIKE_RE.match(lowered):
        return False
    return False


def _looks_like_answer(value: str) -> bool:
    lowered = _normalize_intent_text(value)
    if not lowered:
        return False
    if QUESTION_LIKE_RE.match(lowered):
        return False
    return ANSWER_LIKE_RE.match(lowered) is not None


def _looks_like_filler_fragment(value: str) -> bool:
    normalized = _normalize_intent_text(value)
    if not normalized or QUESTION_LIKE_RE.match(normalized):
        return False

    tokens = _tokenize_question(normalized)
    if len(tokens) < 4:
        return False

    filler_count = sum(1 for token in tokens if FILLER_TOKEN_RE.match(token))
    content_count = len(tokens) - filler_count
    if content_count <= 0:
        return True

    filler_ratio = filler_count / len(tokens)
    return filler_ratio >= 0.45 and content_count <= 3


def _deduplicate_preserving_order(
    values: list[QuestionCandidate],
) -> list[QuestionCandidate]:
    seen_exact: set[str] = set()
    seen_canonical: list[str] = []
    result: list[QuestionCandidate] = []

    for value in values:
        canonical = _canonicalize_question(value.text)
        if canonical in seen_exact:
            continue
        if any(_questions_are_similar(canonical, existing) for existing in seen_canonical):
            continue
        seen_exact.add(canonical)
        seen_canonical.append(canonical)
        result.append(value)

    return result


def _canonicalize_question(value: str) -> str:
    normalized = CANONICAL_WHITESPACE_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")


def _normalize_intent_text(value: str) -> str:
    normalized = normalize_whitespace(value).lower()
    normalized = LEADING_FILLER_RE.sub("", normalized)
    return normalized.strip(" .,!;:-?")


def _questions_are_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False

    similarity = SequenceMatcher(None, left, right).ratio()
    if similarity >= 0.92:
        return True

    left_tokens = set(_tokenize_question(left))
    right_tokens = set(_tokenize_question(right))
    if not left_tokens or not right_tokens:
        return False

    shared = left_tokens & right_tokens
    if not shared:
        return False

    token_overlap = len(shared) / max(len(left_tokens), len(right_tokens))
    token_coverage = len(shared) / min(len(left_tokens), len(right_tokens))

    if similarity >= 0.84 and token_overlap >= 0.75:
        return True
    if token_coverage >= 0.9 and abs(len(left_tokens) - len(right_tokens)) <= 1:
        return True

    return False


def _tokenize_question(value: str) -> list[str]:
    return QUESTION_TOKEN_RE.findall(value.lower())


def _extract_speaker_turns(
    text: str,
    *,
    interviewer_labels: tuple[str, ...],
) -> list[SpeakerTurn]:
    explicit_labels = _normalize_explicit_speaker_labels(interviewer_labels)
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
            previous_turn = turns[-1]
            turns[-1] = SpeakerTurn(
                label=previous_turn.label,
                normalized_label=previous_turn.normalized_label,
                text=f"{previous_turn.text}\n{line}".strip(),
            )

    return [turn for turn in turns if turn.text]


def _resolve_interviewer_labels(
    turns: list[SpeakerTurn],
    *,
    interviewer_labels: tuple[str, ...],
) -> set[str]:
    if not turns:
        return set()

    available_labels = {turn.normalized_label for turn in turns}
    explicit_labels = _normalize_explicit_speaker_labels(interviewer_labels)
    if explicit_labels:
        return available_labels & explicit_labels

    detected_labels = {
        label
        for label in available_labels
        if _label_matches_any_keywords(label, KNOWN_INTERVIEWER_LABELS)
    }
    if detected_labels:
        return detected_labels

    candidate_labels = {
        label
        for label in available_labels
        if _label_matches_any_keywords(label, KNOWN_CANDIDATE_LABELS)
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


def _normalize_explicit_speaker_labels(labels: tuple[str, ...]) -> set[str]:
    return {
        normalized
        for label in labels
        if (normalized := _normalize_speaker_label(label))
    }


def _label_matches_any_keywords(label: str, keywords: tuple[str, ...]) -> bool:
    return any(_label_matches_keyword(label, keyword) for keyword in keywords)


def _label_matches_keyword(label: str, keyword: str) -> bool:
    label_tokens = _tokenize_question(label)
    keyword_tokens = _tokenize_question(keyword)
    if not label_tokens or not keyword_tokens:
        return False

    if len(keyword_tokens) == 1:
        return keyword_tokens[0] in label_tokens

    window_size = len(keyword_tokens)
    for index in range(len(label_tokens) - window_size + 1):
        if label_tokens[index : index + window_size] == keyword_tokens:
            return True
    return False


def _looks_like_known_speaker_label(label: str) -> bool:
    if not label:
        return False
    if KNOWN_SPEAKER_LABEL_RE.match(label):
        return True
    if _label_matches_any_keywords(label, KNOWN_INTERVIEWER_LABELS):
        return True
    if _label_matches_any_keywords(label, KNOWN_CANDIDATE_LABELS):
        return True
    return False


def _ordered_unique(values) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)

    return ordered
