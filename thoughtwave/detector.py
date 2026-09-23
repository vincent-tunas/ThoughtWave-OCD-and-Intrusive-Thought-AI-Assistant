from __future__ import annotations

import re
from dataclasses import dataclass

from .models import BehavioralState, DetectionResult, RelatedMessage


CERTAINTY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("are you sure", re.compile(r"\b(are|r)\s+(you|u)\s+sure\b", re.I)),
    ("how do I know", re.compile(r"\bhow\s+(?:can|do)\s+i\s+know\b", re.I)),
    ("guarantee", re.compile(r"\bguarantee(?:d|s)?\b", re.I)),
    ("100 percent", re.compile(r"\b100\s*(?:%|percent)\b", re.I)),
    ("what if", re.compile(r"\bwhat\s+if\b", re.I)),
    ("certain", re.compile(r"\b(?:completely|absolutely|totally)?\s*certain(?:ty)?\b", re.I)),
    ("prove", re.compile(r"\b(?:prove|proof)\b", re.I)),
    ("does this mean", re.compile(r"\bdoes\s+this\s+mean\b", re.I)),
)

ESCALATION_PATTERN = re.compile(
    r"\b(?:please\s+just\s+tell\s+me|i\s+need\s+to\s+know|answer\s+me|"
    r"still\s+(?:worried|scared|anxious)|can'?t\s+stop|again\s*$)\b",
    re.I,
)

# This is intentionally high precision and incomplete. It is a prompt-level MVP
# override, not a clinical crisis classifier.
URGENT_SAFETY_PATTERN = re.compile(
    r"\b(?:i\s+(?:am|'m)\s+going\s+to\s+(?:kill|hurt)\s+(?:myself|someone)|"
    r"i\s+will\s+(?:kill|hurt)\s+(?:myself|someone)|"
    r"i\s+have\s+a\s+plan\s+to\s+(?:kill|hurt)\s+(?:myself|someone)|"
    r"immediate\s+danger)\b",
    re.I,
)


@dataclass(frozen=True)
class DetectorConfig:
    similarity_threshold: float = 0.72


class BehavioralStateDetector:
    def __init__(self, config: DetectorConfig):
        self.config = config

    def analyze(
        self,
        current_message: str,
        related_messages: list[RelatedMessage],
    ) -> DetectionResult:
        matched_phrases = tuple(
            label for label, pattern in CERTAINTY_PATTERNS if pattern.search(current_message)
        )
        certainty_signal = bool(matched_phrases)
        urgent_safety_signal = bool(URGENT_SAFETY_PATTERN.search(current_message))
        highest_similarity = related_messages[0].similarity if related_messages else 0.0
        qualifying = [
            message
            for message in related_messages
            if message.similarity >= self.config.similarity_threshold
        ]
        related_count = len(qualifying)
        escalation_signal = bool(ESCALATION_PATTERN.search(current_message)) and related_count > 0

        score = 0
        if certainty_signal:
            score += 1
        if highest_similarity >= self.config.similarity_threshold:
            score += 2
        if related_count >= 2:
            score += 2
        if related_count >= 4:
            score += 1
        if escalation_signal:
            score += 1

        if score <= 1:
            state = BehavioralState.NORMAL
        elif score <= 3:
            state = BehavioralState.MONITOR
        elif score <= 5:
            state = BehavioralState.POSSIBLE_LOOP
        else:
            state = BehavioralState.STRONG_LOOP

        return DetectionResult(
            state=state,
            score=score,
            highest_similarity=highest_similarity,
            related_count=related_count,
            certainty_signal=certainty_signal,
            escalation_signal=escalation_signal,
            urgent_safety_signal=urgent_safety_signal,
            matched_certainty_phrases=matched_phrases,
            related_messages=tuple(qualifying),
        )

