from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings
from app.providers.base import BaseLLMProvider, LLMResult, LLMStreamChunk


class OllamaProvider(BaseLLMProvider):
    """Provider that calls a local Ollama instance via its OpenAI-compatible API."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> LLMResult:
        url = f"{self.base_url}/v1/chat/completions"
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature

        try:
            async with httpx.AsyncClient(timeout=120, transport=httpx.AsyncHTTPTransport()) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Ollama request timed out. The model may still be loading."
            ) from e
        except httpx.HTTPStatusError as e:
            detail = f"Ollama returned HTTP {e.response.status_code}"
            try:
                body_err = e.response.json()
                if "error" in body_err:
                    detail += f" - {body_err['error']}"
            except Exception:
                pass
            raise RuntimeError(detail) from e

        choice = data["choices"][0]
        usage = data.get("usage", {})

        return LLMResult(
            content=choice["message"]["content"],
            role=choice["message"].get("role", "assistant"),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=data.get("model", model),
            created=data.get("created", 0),
        )

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        url = f"{self.base_url}/v1/chat/completions"
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature

        try:
            async with httpx.AsyncClient(timeout=120, transport=httpx.AsyncHTTPTransport()) as client:
                async with client.stream("POST", url, json=body) as resp:
                    resp.raise_for_status()
                    async for line in resp.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        payload = line[6:]
                        if payload == "[DONE]":
                            continue
                        chunk = _parse_chunk(payload)
                        if chunk is not None:
                            yield chunk
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. Is Ollama running?"
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError(
                "Ollama request timed out. The model may still be loading."
            ) from e
        except httpx.HTTPStatusError as e:
            detail = f"Ollama returned HTTP {e.response.status_code}"
            try:
                body_err = e.response.json()
                if "error" in body_err:
                    detail += f" - {body_err['error']}"
            except Exception:
                pass
            raise RuntimeError(detail) from e


def _parse_chunk(payload: str) -> LLMStreamChunk | None:
    """Parse a single SSE ``data:`` payload into an ``LLMStreamChunk``."""
    import json

    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        return None

    choices = data.get("choices")
    if not choices:
        return None

    choice = choices[0]
    delta = choice.get("delta", {})

    return LLMStreamChunk(
        content=delta.get("content", ""),
        finish_reason=choice.get("finish_reason"),
    )
