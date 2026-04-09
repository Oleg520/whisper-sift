from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TranscriptionSegment:
    id: int
    start: float
    end: float
    text: str


@dataclass(slots=True, frozen=True)
class TranscriptionDocument:
    text: str
    language: str | None = None
    segments: tuple[TranscriptionSegment, ...] = ()

    @property
    def segment_count(self) -> int:
        return len(self.segments)
