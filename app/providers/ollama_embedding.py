from collections.abc import Sequence

import httpx

from app.core.config import settings
from app.providers.base import BaseEmbeddingProvider


class OllamaEmbeddingProvider(BaseEmbeddingProvider):
    """Generates embeddings via Ollama's ``/api/embed`` endpoint."""

    def __init__(self) -> None:
        self.base_url = settings.ollama_base_url.rstrip("/")
        self.model = settings.embedding_model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self.base_url}/api/embed"
        body = {"model": self.model, "input": list(texts)}

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(url, json=body)
                resp.raise_for_status()
                data = resp.json()
        except httpx.ConnectError as e:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url} for embedding."
            ) from e
        except httpx.TimeoutException as e:
            raise TimeoutError("Ollama embedding request timed out.") from e
        except httpx.HTTPStatusError as e:
            detail = f"Ollama embedding returned HTTP {e.response.status_code}"
            try:
                body_err = e.response.json()
                if "error" in body_err:
                    detail += f" - {body_err['error']}"
            except Exception:
                pass
            raise RuntimeError(detail) from e

        return data.get("embeddings", [])
