from __future__ import annotations

import re
from difflib import SequenceMatcher

from whisper_sift.domain.questions import QuestionCandidate


WHITESPACE_RE = re.compile(r"[ \t]+")
CANONICAL_WHITESPACE_RE = re.compile(r"\s+")
QUESTION_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
LEADING_FILLER_RE = re.compile(
    r"^(?:(?:а|ну|вот|так|и|слушай|смотри)\b\s*[,:\-]?\s*)+",
    re.IGNORECASE,
)


def normalize_whitespace(value: str) -> str:
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = WHITESPACE_RE.sub(" ", value)
    value = re.sub(r" ?\n ?", "\n", value)
    return value.strip()


def deduplicate_preserving_order(
    values: list[QuestionCandidate],
) -> list[QuestionCandidate]:
    seen_exact: set[str] = set()
    seen_canonical: list[str] = []
    result: list[QuestionCandidate] = []

    for value in values:
        canonical = canonicalize_question(value.text)
        if canonical in seen_exact:
            continue
        if any(questions_are_similar(canonical, existing) for existing in seen_canonical):
            continue
        seen_exact.add(canonical)
        seen_canonical.append(canonical)
        result.append(value)

    return result


def canonicalize_question(value: str) -> str:
    normalized = CANONICAL_WHITESPACE_RE.sub(" ", value.lower()).strip()
    return normalized.strip(" .,!;:-?")


def normalize_intent_text(value: str) -> str:
    normalized = normalize_whitespace(value).lower()
    normalized = LEADING_FILLER_RE.sub("", normalized)
    return normalized.strip(" .,!;:-?")


def questions_are_similar(left: str, right: str) -> bool:
    if left == right:
        return True
    if not left or not right:
        return False

    similarity = SequenceMatcher(None, left, right).ratio()
    if similarity >= 0.92:
        return True

    left_tokens = set(tokenize_question(left))
    right_tokens = set(tokenize_question(right))
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


def tokenize_question(value: str) -> list[str]:
    return QUESTION_TOKEN_RE.findall(value.lower())
