from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TranscriptArtifact:
    text: str
    source_name: str | None = None


@dataclass(slots=True, frozen=True)
class SpeakerTurn:
    label: str
    normalized_label: str
    text: str


@dataclass(slots=True, frozen=True)
class TranscriptSlice:
    text: str
    speaker_label: str | None = None
    normalized_speaker_label: str | None = None


@dataclass(slots=True, frozen=True)
class TranscriptView:
    artifact: TranscriptArtifact
    speaker_turns: tuple[SpeakerTurn, ...]
    candidate_slices: tuple[TranscriptSlice, ...]
    selected_interviewer_labels: tuple[str, ...]
    available_speaker_labels: tuple[str, ...]

    @property
    def has_speaker_structure(self) -> bool:
        return bool(self.speaker_turns)

