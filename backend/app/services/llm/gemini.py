"""Native Gemini streaming adapter for the existing text/voice service."""
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import httpx


def _http_client(key: str, timeout: float) -> httpx.AsyncClient:
    kwargs = {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/",
        "headers": {"x-goog-api-key": key},
        "limits": httpx.Limits(max_connections=50, max_keepalive_connections=20, keepalive_expiry=120),
        "timeout": timeout,
    }
    try:
        return httpx.AsyncClient(**kwargs, http2=True)
    except ImportError:
        return httpx.AsyncClient(**kwargs)


class GeminiClient:
    def __init__(self, key, model, http_client=None, timeout=30):
        if not key:
            raise RuntimeError("GEMINI_API_KEY is missing")
        self.key, self.model, self.timeout = key, model, timeout
        self.http = http_client or _http_client(key, timeout)
        self.messages = self
        self.models = self

    def with_options(self, timeout=30, max_retries=0):
        return GeminiClient(self.key, self.model, self.http, timeout)

    async def close(self):
        await self.http.aclose()

    async def list(self, limit=1):
        response = await self.http.get("models", params={"pageSize": limit}, timeout=self.timeout)
        response.raise_for_status()

    def payload(self, system, messages, max_tokens):
        # Gemini 3.x ignores thinkingBudget and thinks until the first answer token.
        # thinkingLevel=minimal is the low-latency setting; mixing both params 400s.
        if str(self.model).startswith("gemini-3"):
            thinking = {"thinkingLevel": "minimal"}
        else:
            thinking = {"thinkingBudget": 0}
        return {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "model" if m["role"] == "assistant" else "user",
                          "parts": [{"text": m["content"]}]} for m in messages],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.2,
                "thinkingConfig": thinking,
            },
        }

    @staticmethod
    def text(data):
        return "".join(part.get("text", "") for candidate in data.get("candidates", [])[:1]
                       for part in candidate.get("content", {}).get("parts", []) if not part.get("thought"))

    @asynccontextmanager
    async def stream(self, *, model, system, messages, max_tokens):
        async with self.http.stream(
            "POST", f"models/{self.model}:streamGenerateContent", params={"alt": "sse"},
            json=self.payload(system, messages, max_tokens), timeout=self.timeout,
        ) as response:
            response.raise_for_status()
            async def tokens():
                received = False
                async for line in response.aiter_lines():
                    if line.startswith("data:"):
                        data = json.loads(line[5:].strip())
                        if "error" in data:
                            raise RuntimeError("Gemini streaming request failed")
                        text = self.text(data)
                        if text:
                            received = True
                            yield text
                if not received:
                    raise RuntimeError("Gemini returned no speech text")
            yield SimpleNamespace(text_stream=tokens())

    async def create(self, **kwargs):
        async with self.stream(**kwargs) as stream:
            text = "".join([part async for part in stream.text_stream])
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)])
