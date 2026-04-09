from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class TranscriptArtifact:
    text: str
    source_name: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "source_name": self.source_name,
            "text_length": len(self.text),
        }


@dataclass(slots=True, frozen=True)
class SpeakerTurn:
    label: str
    normalized_label: str
    text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "normalized_label": self.normalized_label,
            "text": self.text,
        }


@dataclass(slots=True, frozen=True)
class TranscriptSlice:
    text: str
    speaker_label: str | None = None
    normalized_speaker_label: str | None = None
    start_time: str | None = None
    end_time: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "speaker_label": self.speaker_label,
            "normalized_speaker_label": self.normalized_speaker_label,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


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

    def to_dict(self) -> dict[str, object]:
        return {
            "artifact": self.artifact.to_dict(),
            "has_speaker_structure": self.has_speaker_structure,
            "available_speaker_labels": list(self.available_speaker_labels),
            "selected_interviewer_labels": list(self.selected_interviewer_labels),
            "speaker_turn_count": len(self.speaker_turns),
            "candidate_slice_count": len(self.candidate_slices),
            "candidate_slices": [candidate.to_dict() for candidate in self.candidate_slices],
            "speaker_turns": [turn.to_dict() for turn in self.speaker_turns],
        }
