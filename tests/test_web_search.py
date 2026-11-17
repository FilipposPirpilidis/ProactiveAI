import json

import httpx
import pytest

from app.web_search import WebSearchClient, WebSearchError


async def test_web_search_is_bounded_authenticated_and_cached() -> None:
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        assert request.headers["authorization"] == "Bearer secret"
        assert json.loads(request.content) == {
            "query": "latest Swift version",
            "max_results": 2,
        }
        return httpx.Response(
            200,
            json={
                "results": [
                    {
                        "title": "Swift releases",
                        "url": "https://swift.org/releases/",
                        "content": "The current stable Swift release is listed here.",
                    }
                ]
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    client = WebSearchClient(
        enabled=True,
        api_key="secret",
        url="https://ollama.com/api/web_search",
        timeout_seconds=1,
        max_results=2,
        cache_ttl_seconds=60,
        client=http_client,
    )

    first = await client.search("latest Swift version")
    second = await client.search(" latest   Swift version ")

    assert first == second
    assert first[0].url == "https://swift.org/releases/"
    assert calls == 1
    await http_client.aclose()


async def test_web_search_requires_both_flag_and_key() -> None:
    client = WebSearchClient(
        enabled=True,
        api_key="",
        url="https://ollama.com/api/web_search",
        timeout_seconds=1,
        max_results=3,
        cache_ttl_seconds=60,
    )

    assert client.available is False
    with pytest.raises(WebSearchError, match="not configured"):
        await client.search("current weather")
    await client.close()
