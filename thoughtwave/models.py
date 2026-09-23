from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BehavioralState(str, Enum):
    NORMAL = "NORMAL"
    MONITOR = "MONITOR"
    POSSIBLE_LOOP = "POSSIBLE_LOOP"
    STRONG_LOOP = "STRONG_LOOP"


@dataclass(frozen=True)
class RelatedMessage:
    message_id: int
    content: str
    similarity: float
    created_at: str


@dataclass(frozen=True)
class DetectionResult:
    state: BehavioralState
    score: int
    highest_similarity: float
    related_count: int
    certainty_signal: bool
    escalation_signal: bool
    urgent_safety_signal: bool
    matched_certainty_phrases: tuple[str, ...] = ()
    related_messages: tuple[RelatedMessage, ...] = ()

    def to_metadata(self) -> dict[str, Any]:
        return {
            "state": self.state.value,
            "score": self.score,
            "highest_similarity": round(self.highest_similarity, 4),
            "related_count": self.related_count,
            "certainty_signal": self.certainty_signal,
            "escalation_signal": self.escalation_signal,
            "urgent_safety_signal": self.urgent_safety_signal,
            "matched_certainty_phrases": list(self.matched_certainty_phrases),
            "related_message_ids": [m.message_id for m in self.related_messages],
        }


@dataclass(frozen=True)
class ChatMessage:
    id: int | None
    session_id: str
    role: str
    content: str
    created_at: str = field(default_factory=utc_now_iso)
    behavioral_state: str | None = None
    detector_score: int | None = None
    detector_metadata: dict[str, Any] | None = None
    provider_request_id: str | None = None


@dataclass(frozen=True)
class GenerationResult:
    content: str
    request_id: str | None
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


@dataclass(frozen=True)
class TurnResult:
    answer: str
    detection: DetectionResult
    generation: GenerationResult

