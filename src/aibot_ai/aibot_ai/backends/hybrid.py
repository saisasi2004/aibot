"""Ordered backends with a circuit breaker: cloud first, local fallback, no delay on a dead network.

Each member is tried in order. A retryable cloud error (429, 5xx) gets one retry after a short
backoff. After `failures` consecutive failures a member is skipped for `cooldown` seconds; then one
call probes it again. ROS-free; `clock` and `sleep` are injectable for tests.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

from .base import BackendError, LLMBackend, PrivacyError, VLMBackend, check_image_source


@dataclass
class CircuitBreaker:
    failures: int = 3
    cooldown: float = 60.0
    clock: Callable[[], float] = time.monotonic
    count: int = 0
    opened_at: float | None = None

    def allow(self) -> bool:
        """Closed, or open with the cooldown over (then one call probes)."""
        return self.opened_at is None or self.clock() - self.opened_at >= self.cooldown

    def success(self) -> None:
        self.count, self.opened_at = 0, None

    def failure(self) -> None:
        self.count += 1
        if self.count >= self.failures:
            self.opened_at = self.clock()


@dataclass
class _Member:
    backend: object
    breaker: CircuitBreaker


@dataclass
class HybridBackend(LLMBackend, VLMBackend):
    backends: list = field(default_factory=list)
    failures: int = 3
    cooldown: float = 60.0
    retry_backoff: float = 0.5
    clock: Callable[[], float] = time.monotonic
    sleep: Callable[[float], None] = time.sleep
    log: Callable[[str], None] = lambda _m: None
    name = "hybrid"

    def __post_init__(self) -> None:
        self.members = [_Member(b, CircuitBreaker(self.failures, self.cooldown, self.clock)) for b in self.backends]
        self.last_label = ""
        self.last_backend = ""

    def _run(self, call: Callable[[object], dict]) -> dict:
        skipped: list[str] = []
        last: BackendError | None = None
        for m in self.members:
            kind = getattr(m.backend, "kind", m.backend.name)
            if not getattr(m.backend, "available", True):
                skipped.append(f"{kind} unavailable")
                continue
            if not m.breaker.allow():
                skipped.append(f"{kind} down")
                continue
            t0 = self.clock()
            for attempt in (1, 2):
                try:
                    out = call(m.backend)
                except PrivacyError:
                    raise
                except BackendError as exc:
                    last = exc
                    if attempt == 1 and getattr(exc, "retryable", False):
                        self.log(f"{kind}: {exc}; retrying once")
                        self.sleep(self.retry_backoff)
                        continue
                    m.breaker.failure()
                    self.log(f"{kind} failed: {exc}")
                    skipped.append(f"{kind} failed")
                    break
                m.breaker.success()
                note = f" ({', '.join(skipped)})" if skipped else ""
                self.last_backend = kind
                self.last_label = f"{kind} {self.clock() - t0:.1f} s{note}"
                return out
        raise last if last is not None else BackendError("no model backend available: " + ", ".join(skipped))

    def generate(self, messages: list[dict], schema: dict | None = None, local: tuple[list[dict], dict | None] | None = None) -> dict:
        """local: a shorter (messages, schema) for local models: a fallback must stay quick, and the
        body sheet for designing gestures cost a small local model 76 s on this CPU."""
        def call(b):
            if local is not None and getattr(b, "kind", "") == "local":
                return b.generate(*local)
            return b.generate(messages, schema)
        return self._run(call)

    def ask(self, image_jpeg: bytes, prompt: str, schema: dict | None = None, source: str = "") -> dict:
        check_image_source(source)
        return self._run(lambda b: b.ask(image_jpeg, prompt, schema, source=source))

    def describe_last(self) -> str:
        return self.last_label

    def cloud_ready(self) -> bool:
        """The next call will go to the cloud first (key present, breaker not open)."""
        return any(getattr(m.backend, "kind", "") == "cloud" and getattr(m.backend, "available", True) and m.breaker.allow()
                   for m in self.members)

    def check(self) -> None:
        """Startup check: at least one member reachable. Cloud is 'available' if the key exists."""
        problems = []
        for m in self.members:
            b = m.backend
            try:
                if hasattr(b, "check"):
                    b.check()
                if getattr(b, "available", True):
                    return
                problems.append(f"{b.name}: key missing")
            except BackendError as exc:
                problems.append(str(exc))
        raise BackendError("; ".join(problems) or "no backends configured")
