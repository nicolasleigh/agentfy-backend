from collections.abc import AsyncGenerator

import httpx

from app.core.config import settings
from app.providers.base import BaseLLMProvider, LLMResult, LLMStreamChunk


class OpenAIProvider(BaseLLMProvider):
    """Provider that calls the OpenAI API (or any OpenAI-compatible endpoint)."""

    def __init__(self) -> None:
        self.base_url = settings.openai_base_url.rstrip("/")
        self.api_key = settings.openai_api_key

    async def chat_completion(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> LLMResult:
        url = f"{self.base_url}/chat/completions"
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": False,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                resp = await client.post(url, json=body, headers=headers)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to {self.base_url}. Check OPENAI_BASE_URL."
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError("OpenAI request timed out.") from e
        except httpx.HTTPStatusError as e:
            detail = f"OpenAI returned HTTP {e.response.status_code}"
            try:
                body_err = e.response.json()
                err = body_err.get("error", {})
                if isinstance(err, dict):
                    detail += f" - {err.get('message', err)}"
                elif isinstance(err, str):
                    detail += f" - {err}"
            except Exception:
                pass
            raise RuntimeError(detail) from e

        choice = data["choices"][0]
        usage = data.get("usage", {})
        message = choice["message"]

        return LLMResult(
            content=message.get("content") or "",
            role=message.get("role", "assistant"),
            finish_reason=choice.get("finish_reason", "stop"),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            model=data.get("model", model),
            created=data.get("created", 0),
            tool_calls=message.get("tool_calls") or [],
        )

    async def chat_completion_stream(
        self,
        model: str,
        messages: list[dict],
        temperature: float | None = None,
        tools: list[dict] | None = None,
    ) -> AsyncGenerator[LLMStreamChunk, None]:
        url = f"{self.base_url}/chat/completions"
        body: dict = {
            "model": model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if tools:
            body["tools"] = tools

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with httpx.AsyncClient(timeout=120) as client:
                async with client.stream("POST", url, json=body, headers=headers) as resp:
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
                f"Cannot connect to {self.base_url}. Check OPENAI_BASE_URL."
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError("OpenAI request timed out.") from e
        except httpx.HTTPStatusError as e:
            detail = f"OpenAI returned HTTP {e.response.status_code}"
            try:
                body_err = e.response.json()
                err = body_err.get("error", {})
                if isinstance(err, dict):
                    detail += f" - {err.get('message', err)}"
                elif isinstance(err, str):
                    detail += f" - {err}"
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
