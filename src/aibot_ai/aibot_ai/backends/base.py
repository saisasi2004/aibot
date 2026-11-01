"""Model backend interfaces: the planner and the VLM service depend only on these."""
from __future__ import annotations

from abc import ABC, abstractmethod


class BackendError(RuntimeError):
    """The model endpoint is down, the model is missing, or it returned unusable output."""


class LLMBackend(ABC):
    name = "llm"

    @abstractmethod
    def generate(self, messages: list[dict], schema: dict | None = None) -> dict:
        """Chat completion constrained to `schema` (JSON Schema); returns the parsed JSON object."""


class VLMBackend(ABC):
    name = "vlm"

    @abstractmethod
    def ask(self, image_jpeg: bytes, prompt: str, schema: dict | None = None) -> dict:
        """Question about one image, answered as JSON matching `schema`."""
