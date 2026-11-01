"""Ollama backend (local, offline): chat + vision with JSON-schema structured output.

Uses only the standard library (urllib) against Ollama's HTTP API on localhost:
  POST /api/chat  {model, messages, format: <json schema>, stream: false, keep_alive, options}
  GET  /api/tags  installed models (checked at startup: a missing tag is reported, not assumed)
`format` with a JSON schema makes Ollama constrain decoding to that schema, so the planner gets
parseable JSON instead of free text. Memory: one model is resident at a time when keep_alive
evicts the other; `keep_alive` keeps the active one warm between calls.
"""
from __future__ import annotations

import base64
import json
import time
import urllib.error
import urllib.request

from .base import BackendError, LLMBackend, VLMBackend, check_image_source


def list_models(url: str, timeout: float = 3.0) -> list[str]:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as r:
            return [m["name"] for m in json.load(r).get("models", [])]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        raise BackendError(f"Ollama is not reachable at {url} ({exc}). Start it with `ollama serve` "
                           "(or `sudo systemctl start ollama`).") from exc


def require_model(url: str, model: str) -> None:
    models = list_models(url)
    if model not in models and f"{model}:latest" not in models:
        raise BackendError(f"Ollama model '{model}' is not installed (have: {', '.join(models) or 'none'}). "
                           f"Run: ollama pull {model}")


class OllamaBackend(LLMBackend, VLMBackend):
    name = "ollama"
    kind = "local"

    def __init__(self, model: str, url: str = "http://localhost:11434", keep_alive: str = "30m",
                 num_ctx: int = 4096, num_predict: int = 512, temperature: float = 0.1, timeout: float = 120.0,
                 think: bool | None = None) -> None:
        """think: False turns off a reasoning model's hidden "thinking" (granite4.2 spends ~240
        tokens, 80+ s on this CPU, before a 15-token JSON answer); None = model default
        (not sent, for models without the option)."""
        self.model, self.url, self.keep_alive, self.think = model, url.rstrip("/"), keep_alive, think
        self.options = {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": temperature}
        self.timeout = timeout
        self.last_stats: dict = {}

    def check(self) -> None:
        require_model(self.url, self.model)

    def _chat(self, messages: list[dict], schema: dict | None) -> dict:
        """Streamed chat; returns the first complete JSON object.

        Why streamed: with a JSON-schema `format`, small models finish the object and then keep
        emitting whitespace until num_predict (measured: a 15-token answer followed by 244
        whitespace tokens, 119 s on a loaded CPU). Reading the stream and closing it as soon as
        the braces balance stops the generation there (Ollama aborts on disconnect)."""
        body = {"model": self.model, "messages": messages, "stream": True, "keep_alive": self.keep_alive,
                "options": self.options}
        if schema is not None:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(f"{self.url}/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        t0 = time.monotonic()
        text, chunks, first_s, final = "", 0, None, {}
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                for line in r:
                    if time.monotonic() - t0 > self.timeout:
                        raise BackendError(f"no complete answer within {self.timeout:.0f} s")
                    if not line.strip():
                        continue
                    part = json.loads(line)
                    if part.get("error"):
                        raise BackendError(f"Ollama error: {part['error']}")
                    piece = part.get("message", {}).get("content", "")
                    if piece:
                        chunks += 1
                        first_s = first_s if first_s is not None else time.monotonic() - t0
                        text += piece
                    if part.get("done"):
                        final = part
                        break
                    if _complete_object(text):
                        break                                    # closing the stream stops generation
        except urllib.error.HTTPError as exc:
            raise BackendError(f"Ollama error {exc.code}: {exc.read().decode(errors='replace')[:300]}") from exc
        except (urllib.error.URLError, OSError) as exc:
            raise BackendError(f"Ollama request failed: {exc}") from exc
        ns = 1e-9
        self.last_stats = {"wall_s": time.monotonic() - t0, "first_token_s": first_s, "output_chunks": chunks,
                           "load_s": final.get("load_duration", 0) * ns, "prompt_tokens": final.get("prompt_eval_count"),
                           "output_tokens": final.get("eval_count", chunks), "eval_s": final.get("eval_duration", 0) * ns}
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise BackendError(f"model returned non-JSON output: {text[:200]!r}") from exc

    def generate(self, messages: list[dict], schema: dict | None = None) -> dict:
        return self._chat(messages, schema)

    @property
    def last_backend(self) -> str:
        return "local"

    def describe_last(self) -> str:
        return f"local {self.last_stats.get('wall_s', 0.0):.1f} s"

    def ask(self, image_jpeg: bytes, prompt: str, schema: dict | None = None, source: str = "") -> dict:
        check_image_source(source)
        msg = {"role": "user", "content": prompt, "images": [base64.b64encode(image_jpeg).decode()]}
        return self._chat([msg], schema)


def _complete_object(text: str) -> bool:
    """True once `text` holds a complete top-level JSON object (braces balanced outside strings)."""
    depth, in_str, esc, started = 0, False, False, False
    for ch in text:
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
            started = True
        elif ch == "}":
            depth -= 1
            if started and depth == 0:
                return True
    return False
