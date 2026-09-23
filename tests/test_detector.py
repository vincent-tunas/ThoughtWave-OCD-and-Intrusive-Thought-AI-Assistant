from thoughtwave.detector import BehavioralStateDetector, DetectorConfig
from thoughtwave.models import BehavioralState, RelatedMessage


def related(count: int, similarity: float = 0.85) -> list[RelatedMessage]:
    return [
        RelatedMessage(i, f"Earlier concern {i}", similarity, "2026-01-01T00:00:00Z")
        for i in range(1, count + 1)
    ]


def test_ordinary_first_question_is_normal():
    detector = BehavioralStateDetector(DetectorConfig(0.72))
    result = detector.analyze("How does solar energy work?", [])
    assert result.state is BehavioralState.NORMAL
    assert result.score == 0


def test_one_similar_message_is_monitor():
    detector = BehavioralStateDetector(DetectorConfig(0.72))
    result = detector.analyze("Could you explain this in another way?", related(1))
    assert result.state is BehavioralState.MONITOR
    assert result.score == 2


def test_repetition_reaches_possible_loop():
    detector = BehavioralStateDetector(DetectorConfig(0.72))
    result = detector.analyze("Are you sure this does not define me?", related(2))
    assert result.state is BehavioralState.POSSIBLE_LOOP
    assert result.score == 5


def test_sustained_repetition_reaches_strong_loop():
    detector = BehavioralStateDetector(DetectorConfig(0.72))
    result = detector.analyze("Can you guarantee this 100 percent?", related(4))
    assert result.state is BehavioralState.STRONG_LOOP
    assert result.score == 6


def test_intrusive_harm_wording_alone_does_not_trigger_immediate_danger():
    detector = BehavioralStateDetector(DetectorConfig(0.72))
    result = detector.analyze("What if I have an unwanted thought about harm?", [])
    assert result.urgent_safety_signal is False

