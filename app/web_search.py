import asyncio
import time
from dataclasses import dataclass

import httpx


class WebSearchError(RuntimeError):
    pass


@dataclass(frozen=True)
class WebSearchResult:
    title: str
    url: str
    content: str


class WebSearchClient:
    """Small optional Ollama web-search client with a bounded in-memory cache."""

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        url: str,
        timeout_seconds: float,
        max_results: int,
        cache_ttl_seconds: int,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.available = enabled and bool(api_key.strip())
        self._api_key = api_key.strip()
        self.url = url
        self.max_results = max_results
        self.cache_ttl_seconds = cache_ttl_seconds
        self._cache: dict[str, tuple[float, tuple[WebSearchResult, ...]]] = {}
        self._lock = asyncio.Lock()
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=timeout_seconds,
            headers={"Authorization": f"Bearer {api_key.strip()}"},
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def search(self, query: str) -> tuple[WebSearchResult, ...]:
        if not self.available:
            raise WebSearchError("Web search is not configured")
        normalized = " ".join(query.split())[:600]
        if not normalized:
            raise WebSearchError("Web search query is empty")

        cache_key = normalized.casefold()
        now = time.monotonic()
        cached = self._cache.get(cache_key)
        if cached and now - cached[0] <= self.cache_ttl_seconds:
            return cached[1]

        async with self._lock:
            cached = self._cache.get(cache_key)
            now = time.monotonic()
            if cached and now - cached[0] <= self.cache_ttl_seconds:
                return cached[1]
            try:
                response = await self._client.post(
                    self.url,
                    headers={"Authorization": f"Bearer {self._api_key}"},
                    json={"query": normalized, "max_results": self.max_results},
                )
                response.raise_for_status()
                payload = response.json()
                raw_results = payload.get("results", [])
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                raise WebSearchError("Web search request failed") from exc

            results: list[WebSearchResult] = []
            for item in raw_results[: self.max_results]:
                if not isinstance(item, dict):
                    continue
                title = " ".join(str(item.get("title", "")).split())
                url = str(item.get("url", "")).strip()
                content = " ".join(str(item.get("content", "")).split())
                if url and content:
                    results.append(
                        WebSearchResult(
                            title=title[:300],
                            url=url[:2_000],
                            content=content[:1_500],
                        )
                    )
            if not results:
                raise WebSearchError("Web search returned no usable results")
            final = tuple(results)
            self._cache[cache_key] = (time.monotonic(), final)
            return final
