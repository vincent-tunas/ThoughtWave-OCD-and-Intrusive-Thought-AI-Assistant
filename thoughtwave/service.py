from __future__ import annotations

from .database import Database
from .detector import BehavioralStateDetector
from .embeddings import SemanticMemory
from .models import TurnResult, utc_now_iso
from .openrouter import OpenRouterClient
from .prompts import build_messages


class ThoughtWaveService:
    def __init__(
        self,
        database: Database,
        semantic_memory: SemanticMemory,
        detector: BehavioralStateDetector,
        llm: OpenRouterClient,
        top_k: int = 5,
        recent_history_limit: int = 10,
    ):
        self.database = database
        self.semantic_memory = semantic_memory
        self.detector = detector
        self.llm = llm
        self.top_k = top_k
        self.recent_history_limit = recent_history_limit

    def respond(self, session_id: str, user_text: str) -> TurnResult:
        cleaned_text = " ".join(user_text.strip().split())
        if not cleaned_text:
            raise ValueError("Please enter a message.")
        if len(cleaned_text) > 8000:
            raise ValueError("The message is too long. Keep it under 8,000 characters.")

        self.database.ensure_session(session_id)
        # Query before insertion so the current message can never match itself.
        current_vector, related = self.semantic_memory.query(
            session_id=session_id,
            text=cleaned_text,
            top_k=self.top_k,
        )
        detection = self.detector.analyze(cleaned_text, related)
        history = self.database.get_messages(
            session_id, limit=self.recent_history_limit
        )
        messages = build_messages(cleaned_text, detection, history)

        # There is deliberately no local answer fallback. A displayed assistant
        # answer must come from a validated OpenRouter completion.
        generation = self.llm.generate(messages)

        user_message_id, _ = self.database.save_successful_turn(
            session_id=session_id,
            user_text=cleaned_text,
            assistant_text=generation.content,
            detection=detection,
            provider_request_id=generation.request_id,
        )
        self.semantic_memory.add(
            message_id=user_message_id,
            session_id=session_id,
            content=cleaned_text,
            created_at=utc_now_iso(),
            vector=current_vector,
        )
        return TurnResult(
            answer=generation.content,
            detection=detection,
            generation=generation,
        )

