import json
import logging
from collections.abc import Sequence

import httpx

logger = logging.getLogger(__name__)


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, base_url: str, model: str, timeout: float = 45.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=timeout)

    async def close(self) -> None:
        await self._client.aclose()

    async def health(self) -> bool:
        try:
            response = await self._client.get(f"{self.base_url}/api/tags", timeout=3)
            return response.is_success
        except httpx.HTTPError:
            return False

    async def chat(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float = 0.2,
        json_output: bool = False,
    ) -> str:
        payload: dict[str, object] = {
            "model": self.model,
            "messages": list(messages),
            "stream": False,
            "think": False,
            "options": {"temperature": temperature},
        }
        if json_output:
            payload["format"] = "json"
        try:
            response = await self._client.post(f"{self.base_url}/api/chat", json=payload)
            response.raise_for_status()
            content = response.json()["message"]["content"]
            return str(content).strip()
        except (httpx.HTTPError, KeyError, ValueError, json.JSONDecodeError) as exc:
            logger.warning("Ollama request failed: %s", exc)
            raise OllamaError("The local language model is unavailable") from exc
