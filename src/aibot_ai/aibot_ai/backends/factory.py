"""Build the configured backend for a role from plain settings (ai.yaml values)."""
from __future__ import annotations

from typing import Callable

from .hybrid import HybridBackend
from .ollama import OllamaBackend
from .openrouter import OpenRouterBackend
from .secrets import api_key, key_status

MODES = ("hybrid", "openrouter", "ollama")


def build(mode: str, cloud: dict, local: dict, breaker: dict, log: Callable[[str], None] = lambda _m: None):
    """mode: hybrid | openrouter | ollama. Without the key, hybrid degrades to local with one notice."""
    if mode not in MODES:
        raise SystemExit(f"backend '{mode}' not supported; use one of {', '.join(MODES)}")
    log(key_status())
    remote = OpenRouterBackend(**cloud)
    near = OllamaBackend(**local)
    if mode == "openrouter":
        return remote
    if mode == "ollama":
        return near
    if not api_key():
        log("no OpenRouter key: using local models only (set OPENROUTER_API_KEY or fill .env for the cloud)")
    return HybridBackend([remote, near], breaker.get("failures", 3), breaker.get("cooldown", 60.0),
                         breaker.get("retry_backoff", 0.5), log=log)
