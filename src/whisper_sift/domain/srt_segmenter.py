from __future__ import annotations

import re
from collections.abc import Callable

from whisper_sift.domain.transcript import TranscriptSlice

SRT_TIMECODE_RE = re.compile(
    r"^(?P<start>\d{2}:\d{2}:\d{2},\d{3})\s+-->\s+"
    r"(?P<end>\d{2}:\d{2}:\d{2},\d{3})(?:\s+.*)?$"
)


def extract_srt_slices(
    text: str,
    *,
    normalize_whitespace: Callable[[str], str],
    cleanup_question: Callable[[str], str],
    looks_like_question_without_mark: Callable[[str], bool],
    looks_like_answer: Callable[[str], bool],
    split_speaker_labeled_text: Callable[[str], tuple[str, str, str] | None],
) -> list[TranscriptSlice]:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_blocks = [block for block in normalized.split("\n\n") if block.strip()]

    timed_blocks: list[TranscriptSlice] = []
    for raw_block in raw_blocks:
        lines = [line.strip() for line in raw_block.splitlines() if line.strip()]
        if len(lines) < 2:
            continue

        if lines[0].isdigit():
            timecode_line = lines[1]
            body_lines = lines[2:]
        else:
            timecode_line = lines[0]
            body_lines = lines[1:]

        match = SRT_TIMECODE_RE.match(timecode_line)
        if match is None or not body_lines:
            continue

        body = normalize_whitespace("\n".join(body_lines))
        if not body:
            continue

        timed_blocks.append(
            TranscriptSlice(
                text=body,
                start_time=match.group("start"),
                end_time=match.group("end"),
            )
        )

    if not timed_blocks:
        return []

    collapsed_slices: list[TranscriptSlice] = []
    text_parts: list[str] = []
    start_time: str | None = None
    end_time: str | None = None

    for index, block in enumerate(timed_blocks):
        if not text_parts:
            start_time = block.start_time
        text_parts.append(block.text)
        end_time = block.end_time
        buffered_text = normalize_whitespace(" ".join(text_parts))
        next_block = timed_blocks[index + 1] if index + 1 < len(timed_blocks) else None

        if _should_close_srt_slice(
            buffered_text=buffered_text,
            current_block_text=block.text,
            next_block=next_block,
            cleanup_question=cleanup_question,
            looks_like_question_without_mark=looks_like_question_without_mark,
            looks_like_answer=looks_like_answer,
            split_speaker_labeled_text=split_speaker_labeled_text,
        ):
            collapsed_slices.append(
                TranscriptSlice(
                    text=buffered_text,
                    start_time=start_time,
                    end_time=end_time,
                )
            )
            text_parts = []
            start_time = None
            end_time = None

    if text_parts:
        collapsed_slices.append(
            TranscriptSlice(
                text=normalize_whitespace(" ".join(text_parts)),
                start_time=start_time,
                end_time=end_time,
            )
        )

    return collapsed_slices


def _ends_sentence(value: str) -> bool:
    return bool(re.search(r"[?.!…]\s*$", value))


def _should_close_srt_slice(
    *,
    buffered_text: str,
    current_block_text: str,
    next_block: TranscriptSlice | None,
    cleanup_question: Callable[[str], str],
    looks_like_question_without_mark: Callable[[str], bool],
    looks_like_answer: Callable[[str], bool],
    split_speaker_labeled_text: Callable[[str], tuple[str, str, str] | None],
) -> bool:
    if _ends_sentence(current_block_text):
        return True
    if next_block is None:
        return True

    current_speaker = split_speaker_labeled_text(buffered_text)
    next_speaker = split_speaker_labeled_text(next_block.text)
    if current_speaker is not None and next_speaker is not None:
        return True

    cleaned_buffer = cleanup_question(buffered_text)
    if not cleaned_buffer or not looks_like_question_without_mark(cleaned_buffer):
        return False

    next_text = next_speaker[2] if next_speaker is not None else next_block.text
    next_cleaned = cleanup_question(next_text)
    if not next_cleaned:
        return False
    return looks_like_answer(next_cleaned)
