from pathlib import Path

import numpy as np
import pytest

from thoughtwave.database import Database
from thoughtwave.detector import BehavioralStateDetector, DetectorConfig
from thoughtwave.embeddings import SemanticMemory
from thoughtwave.models import GenerationResult
from thoughtwave.openrouter import OpenRouterError
from thoughtwave.service import ThoughtWaveService


class FakeEncoder:
    def encode(self, texts):
        vectors = []
        for text in texts:
            lowered = text.lower()
            if "bad person" in lowered or "define me" in lowered:
                vector = [1.0, 0.0]
            else:
                vector = [0.0, 1.0]
            vectors.append(vector)
        return np.asarray(vectors, dtype=np.float32)


class RecordingLLM:
    def __init__(self):
        self.calls = []

    def generate(self, messages):
        self.calls.append(messages)
        question = messages[-1]["content"]
        return GenerationResult(
            content=f"Generated for: {question}", request_id="req_1", model="fake/model"
        )


class FailingLLM:
    def generate(self, messages):
        raise OpenRouterError("OpenRouter failed")


def make_service(path: Path, llm):
    database = Database(path)
    database.initialize()
    memory = SemanticMemory(FakeEncoder())
    memory.rebuild(database)
    service = ThoughtWaveService(
        database,
        memory,
        BehavioralStateDetector(DetectorConfig(0.72)),
        llm,
    )
    return database, service


def test_service_sends_the_exact_current_question_to_llm(tmp_path):
    llm = RecordingLLM()
    database, service = make_service(tmp_path / "test.db", llm)
    session_id = database.create_session()
    result = service.respond(session_id, "Does this thought define me?")
    assert llm.calls[0][-1] == {
        "role": "user",
        "content": "Does this thought define me?",
    }
    assert result.answer == "Generated for: Does this thought define me?"
    assert len(database.get_messages(session_id)) == 2


def test_failed_llm_call_does_not_create_a_fake_assistant_message(tmp_path):
    database, service = make_service(tmp_path / "test.db", FailingLLM())
    session_id = database.create_session()
    with pytest.raises(OpenRouterError):
        service.respond(session_id, "Please answer me")
    assert database.get_messages(session_id) == []


def test_repeated_conversation_moves_through_all_four_states(tmp_path):
    llm = RecordingLLM()
    database, service = make_service(tmp_path / "test.db", llm)
    session_id = database.create_session()

    results = [
        service.respond(session_id, "Does this mean I am a bad person?"),
        service.respond(session_id, "Could this thought define me?"),
        service.respond(session_id, "Are you sure I am not a bad person?"),
        service.respond(session_id, "How do I know this does not define me?"),
        service.respond(session_id, "Can you guarantee this does not make me a bad person?"),
    ]

    states = [result.detection.state.value for result in results]
    assert states == [
        "NORMAL",
        "MONITOR",
        "POSSIBLE_LOOP",
        "POSSIBLE_LOOP",
        "STRONG_LOOP",
    ]
