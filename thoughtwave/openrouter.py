from __future__ import annotations

import time
from typing import Any

import httpx

from .models import GenerationResult


class OpenRouterError(RuntimeError):
    """A user-safe OpenRouter failure with no secret or message leakage."""


class OpenRouterClient:
    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 45.0,
        max_retries: int = 2,
        app_url: str = "http://localhost:8501",
        transport: httpx.BaseTransport | None = None,
    ):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is missing. Add it to your .env file.")
        if not model:
            raise ValueError("OPENROUTER_MODEL is missing. Add it to your .env file.")
        self.model = model
        self.max_retries = max(0, max_retries)
        self._client = httpx.Client(
            # Keep a trailing slash and use a relative request path below. A
            # leading slash would otherwise drop the /api/v1 prefix.
            base_url=base_url.rstrip("/") + "/",
            timeout=timeout_seconds,
            transport=transport,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": app_url,
                "X-Title": "ThoughtWave",
            },
        )

    def generate(self, messages: list[dict[str, str]]) -> GenerationResult:
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("The OpenRouter request must end with the current user message.")

        response: httpx.Response | None = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._client.post(
                    "chat/completions",
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": 0.35,
                    },
                )
            except httpx.TimeoutException as exc:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2**attempt))
                    continue
                raise OpenRouterError(
                    "OpenRouter timed out before returning an answer. Please try again."
                ) from exc
            except httpx.RequestError as exc:
                raise OpenRouterError(
                    "ThoughtWave could not connect to OpenRouter. Check your connection and configuration."
                ) from exc

            if response.status_code in {408, 429, 500, 502, 503, 504} and attempt < self.max_retries:
                time.sleep(0.5 * (2**attempt))
                continue
            break

        assert response is not None
        if response.is_error:
            detail = self._error_detail(response)
            if response.status_code == 401:
                raise OpenRouterError("OpenRouter rejected the API key (401).")
            if response.status_code == 402:
                raise OpenRouterError("OpenRouter reported insufficient credits (402).")
            if response.status_code == 404:
                raise OpenRouterError(
                    f"OpenRouter could not access model '{self.model}' (404)."
                )
            if response.status_code == 429:
                raise OpenRouterError("OpenRouter rate-limited the request (429). Try again shortly.")
            raise OpenRouterError(
                f"OpenRouter returned HTTP {response.status_code}: {detail}"
            )

        try:
            payload = response.json()
            choices = payload.get("choices") or []
            content = choices[0]["message"]["content"] if choices else None
            content = self._normalize_content(content)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            raise OpenRouterError("OpenRouter returned a malformed response.") from exc

        if not content.strip():
            raise OpenRouterError("OpenRouter returned an empty answer.")

        usage = payload.get("usage") or {}
        return GenerationResult(
            content=content.strip(),
            request_id=response.headers.get("x-request-id") or payload.get("id"),
            model=str(payload.get("model") or self.model),
            prompt_tokens=self._as_int(usage.get("prompt_tokens")),
            completion_tokens=self._as_int(usage.get("completion_tokens")),
        )

    def close(self) -> None:
        self._client.close()

    @staticmethod
    def _normalize_content(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        return ""

    @staticmethod
    def _error_detail(response: httpx.Response) -> str:
        try:
            payload = response.json()
            error = payload.get("error") or {}
            message = error.get("message") if isinstance(error, dict) else None
            return str(message or "Request failed")[:300]
        except ValueError:
            return "Request failed"

    @staticmethod
    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None
