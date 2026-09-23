from thoughtwave.models import BehavioralState, DetectionResult
from thoughtwave.prompts import build_messages


def detection(state: BehavioralState) -> DetectionResult:
    return DetectionResult(
        state=state,
        score=0,
        highest_similarity=0.0,
        related_count=0,
        certainty_signal=False,
        escalation_signal=False,
        urgent_safety_signal=False,
    )


def test_current_question_is_always_final_user_message():
    messages = build_messages(
        "What should I do today?", detection(BehavioralState.NORMAL), []
    )
    assert messages[-1] == {"role": "user", "content": "What should I do today?"}


def test_each_state_has_distinct_instruction():
    prompts = {
        state: build_messages("Question", detection(state), [])[1]["content"]
        for state in BehavioralState
    }
    assert len(set(prompts.values())) == 4
    assert "do not tell the user" in prompts[BehavioralState.MONITOR].lower()
    assert "tentatively" in prompts[BehavioralState.POSSIBLE_LOOP].lower()
    assert "strong interaction-level evidence" in prompts[BehavioralState.STRONG_LOOP].lower()

