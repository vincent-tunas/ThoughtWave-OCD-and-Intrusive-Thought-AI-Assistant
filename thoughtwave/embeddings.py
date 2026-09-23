from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Protocol, Sequence

import numpy as np

from .database import Database
from .models import ChatMessage, RelatedMessage


class Encoder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEncoder:
    def __init__(self, model_name: str, hf_token: str):
        if not hf_token:
            raise ValueError("HF_TOKEN is missing. Add it to your .env file.")
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "sentence-transformers is not installed. Run pip install -r requirements.txt."
            ) from exc

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, token=hf_token)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        vectors = self._model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)


@dataclass(frozen=True)
class IndexedMessage:
    message_id: int
    session_id: str
    content: str
    created_at: str
    vector: np.ndarray


class SemanticMemory:
    """Thread-safe, session-scoped in-memory semantic index."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder
        self._items: list[IndexedMessage] = []
        self._lock = RLock()

    def rebuild(self, database: Database) -> int:
        messages = database.get_all_user_messages()
        with self._lock:
            self._items = []
            if not messages:
                return 0
            vectors = self.encoder.encode([message.content for message in messages])
            for message, vector in zip(messages, vectors, strict=True):
                if message.id is None:
                    continue
                self._items.append(self._to_indexed(message, vector))
            return len(self._items)

    def query(
        self,
        session_id: str,
        text: str,
        top_k: int,
    ) -> tuple[np.ndarray, list[RelatedMessage]]:
        vector = self.encoder.encode([text])[0]
        with self._lock:
            candidates = [item for item in self._items if item.session_id == session_id]

        if not candidates:
            return vector, []

        matrix = np.vstack([item.vector for item in candidates])
        similarities = matrix @ vector
        ranked_indices = np.argsort(similarities)[::-1][:top_k]
        related = [
            RelatedMessage(
                message_id=candidates[int(index)].message_id,
                content=candidates[int(index)].content,
                similarity=float(similarities[int(index)]),
                created_at=candidates[int(index)].created_at,
            )
            for index in ranked_indices
        ]
        return vector, related

    def add(
        self,
        message_id: int,
        session_id: str,
        content: str,
        created_at: str,
        vector: np.ndarray,
    ) -> None:
        with self._lock:
            self._items.append(
                IndexedMessage(message_id, session_id, content, created_at, vector)
            )

    def remove_session(self, session_id: str) -> None:
        with self._lock:
            self._items = [item for item in self._items if item.session_id != session_id]

    @staticmethod
    def _to_indexed(message: ChatMessage, vector: np.ndarray) -> IndexedMessage:
        assert message.id is not None
        return IndexedMessage(
            message_id=message.id,
            session_id=message.session_id,
            content=message.content,
            created_at=message.created_at,
            vector=np.asarray(vector, dtype=np.float32),
        )

