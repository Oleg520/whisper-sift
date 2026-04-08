from __future__ import annotations

import re


WHITESPACE_RE = re.compile(r"[ \t]+")
QUESTION_SPLIT_RE = re.compile(r"(?<=[?.!])\s+|\n+")


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
) -> list[str]:
    prepared_text = normalize_whitespace(text)
    chunks = QUESTION_SPLIT_RE.split(prepared_text)

    questions: list[str] = []
    for chunk in chunks:
        if "?" not in chunk:
            continue

        for part in chunk.split("?"):
            cleaned = _cleanup_question(part)
            if not cleaned:
                continue
            cleaned += "?"
            if len(cleaned) < min_length or len(cleaned) > max_length:
                continue
            if _looks_like_noise(cleaned):
                continue
            questions.append(cleaned)

    if deduplicate:
        return _deduplicate_preserving_order(questions)
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


def _deduplicate_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result
