"""OpenRouter backend: chat and vision through the OpenAI-compatible API, standard library only.

    POST {url}/chat/completions  stream, response_format json_schema, provider routing, reasoning

Streamed so a real first-byte limit is possible: OpenRouter sends ": OPENROUTER PROCESSING" comments
while waiting, which do not count as an answer. Limits are wall-clock: `first_byte_timeout` until the
first content, `total_timeout` for the whole call. The key comes from OPENROUTER_API_KEY only (see
secrets.py) and is never logged; error messages quote the server's text, not the request.
"""
from __future__ import annotations

import base64
import json
import socket
import time
import urllib.error
import urllib.request

from .base import BackendError, LLMBackend, VLMBackend, check_image_source
from .schema_check import schema_errors
from .secrets import api_key

RETRYABLE_STATUS = (408, 429, 500, 502, 503, 504)


class CloudError(BackendError):
    def __init__(self, message: str, status: int | None = None, retryable: bool = False) -> None:
        super().__init__(message)
        self.status, self.retryable = status, retryable


class OpenRouterBackend(LLMBackend, VLMBackend):
    name = "openrouter"
    kind = "cloud"

    def __init__(self, model: str, url: str = "https://openrouter.ai/api/v1", first_byte_timeout: float = 4.0,
                 total_timeout: float = 15.0, max_tokens: int = 800, temperature: float = 0.0,
                 reasoning_effort: str = "", sort: str = "latency", title: str = "aibot") -> None:
        self.model, self.url = model, url.rstrip("/")
        self.first_byte_timeout, self.total_timeout = first_byte_timeout, total_timeout
        self.max_tokens, self.temperature = max_tokens, temperature
        self.reasoning_effort, self.sort, self.title = reasoning_effort, sort, title
        self.last_stats: dict = {}

    @property
    def available(self) -> bool:
        return bool(api_key())

    def check(self) -> None:
        if not self.available:
            raise CloudError("OPENROUTER_API_KEY is not set")

    def _body(self, messages: list[dict], schema: dict | None) -> dict:
        body: dict = {"model": self.model, "messages": messages, "stream": True, "max_tokens": self.max_tokens,
                      "temperature": self.temperature, "usage": {"include": True},
                      "provider": {"require_parameters": True, "sort": self.sort}}
        if schema is not None:
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "answer", "strict": True, "schema": schema}}
        if self.reasoning_effort:
            body["reasoning"] = {"effort": self.reasoning_effort}
        return body

    def _chat(self, messages: list[dict], schema: dict | None) -> dict:
        key = api_key()
        if not key:
            raise CloudError("OPENROUTER_API_KEY is not set")
        req = urllib.request.Request(f"{self.url}/chat/completions", data=json.dumps(self._body(messages, schema)).encode(),
                                     headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}",
                                              "X-Title": self.title})
        t0 = time.monotonic()
        text, first = "", None
        usage: dict = {}
        provider = ""
        try:
            with urllib.request.urlopen(req, timeout=max(self.first_byte_timeout, 0.1)) as r:
                for raw in r:
                    now = time.monotonic() - t0
                    if now > self.total_timeout:
                        raise CloudError(f"no complete answer within {self.total_timeout:.0f} s", retryable=False)
                    if first is None and now > self.first_byte_timeout:
                        raise CloudError(f"no first byte within {self.first_byte_timeout:.0f} s", retryable=False)
                    line = raw.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue                               # keep-alive comments, blank separators
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    try:
                        chunk = json.loads(payload)
                    except ValueError:
                        continue
                    if chunk.get("error"):
                        err = chunk["error"]
                        status = err.get("code") if isinstance(err.get("code"), int) else None
                        raise CloudError(f"OpenRouter stream error {status}: {str(err.get('message'))[:200]}", status,
                                         status in RETRYABLE_STATUS)
                    provider = chunk.get("provider", provider)
                    usage = chunk.get("usage") or usage
                    for choice in chunk.get("choices", []):
                        piece = (choice.get("delta") or {}).get("content") or ""
                        if piece and first is None:
                            first = time.monotonic() - t0
                        text += piece
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:200]
            raise CloudError(f"OpenRouter HTTP {exc.code}: {detail}", exc.code, exc.code in RETRYABLE_STATUS) from exc
        except (socket.timeout, TimeoutError) as exc:
            raise CloudError(f"OpenRouter timed out after {time.monotonic() - t0:.1f} s") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise CloudError(f"OpenRouter request failed: {exc}") from exc
        self.last_stats = {"wall_s": time.monotonic() - t0, "first_token_s": first, "provider": provider,
                           "output_tokens": usage.get("completion_tokens"), "prompt_tokens": usage.get("prompt_tokens"),
                           "cost": usage.get("cost")}
        try:
            data = json.loads(text)
        except ValueError as exc:
            raise CloudError(f"model returned non-JSON output: {text[:120]!r}") from exc
        if schema is not None and (errs := schema_errors(data, schema)):
            raise CloudError(f"answer does not match the schema: {'; '.join(errs[:3])}")
        return data

    def generate(self, messages: list[dict], schema: dict | None = None) -> dict:
        return self._chat(messages, schema)

    def ask(self, image_jpeg: bytes, prompt: str, schema: dict | None = None, source: str = "") -> dict:
        check_image_source(source)
        url = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode()
        return self._chat([{"role": "user", "content": [{"type": "text", "text": prompt},
                                                        {"type": "image_url", "image_url": {"url": url}}]}], schema)

    def describe_last(self) -> str:
        return f"cloud {self.last_stats.get('wall_s', 0.0):.1f} s"
