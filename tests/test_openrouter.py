import httpx
import pytest

from thoughtwave.openrouter import OpenRouterClient, OpenRouterError


def test_valid_completion_is_returned():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"x-request-id": "req_test"},
            json={
                "id": "gen_test",
                "model": "test/model",
                "choices": [{"message": {"content": "A generated answer."}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 4},
            },
        )

    client = OpenRouterClient(
        "test-key", "test/model", transport=httpx.MockTransport(handler)
    )
    result = client.generate([{"role": "user", "content": "Hello"}])
    assert result.content == "A generated answer."
    assert result.request_id == "req_test"


def test_empty_completion_fails_instead_of_using_fallback():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            200,
            json={"choices": [{"message": {"content": ""}}]},
        )
    )
    client = OpenRouterClient("test-key", "test/model", transport=transport)
    with pytest.raises(OpenRouterError, match="empty answer"):
        client.generate([{"role": "user", "content": "Hello"}])


def test_invalid_key_is_explicit():
    transport = httpx.MockTransport(
        lambda request: httpx.Response(
            401, json={"error": {"message": "Unauthorized"}}
        )
    )
    client = OpenRouterClient(
        "bad-key", "test/model", max_retries=0, transport=transport
    )
    with pytest.raises(OpenRouterError, match="API key"):
        client.generate([{"role": "user", "content": "Hello"}])
