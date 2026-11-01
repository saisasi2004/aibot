"""Model backend interfaces: the planner and the VLM service depend only on these."""
from __future__ import annotations

from abc import ABC, abstractmethod

HEAD_CAMERA = "head_camera"      # the simulated robot camera: the only image source models may see


class BackendError(RuntimeError):
    """The model endpoint is down, the model is missing, or it returned unusable output."""

    retryable = False


class PrivacyError(BackendError):
    """An image that is not from the robot's simulated head camera (e.g. the operator's webcam)."""


def check_image_source(source: str) -> None:
    """Privacy rule, enforced at every VLM entry point: webcam frames show the operator's face and
    must never reach a model or the network. A caller has to name the source explicitly."""
    if source != HEAD_CAMERA:
        raise PrivacyError(f"refusing to send an image from '{source or 'unnamed source'}' to a model: "
                           f"only '{HEAD_CAMERA}' images are allowed")


class LLMBackend(ABC):
    name = "llm"

    @abstractmethod
    def generate(self, messages: list[dict], schema: dict | None = None) -> dict:
        """Chat completion constrained to `schema` (JSON Schema); returns the parsed JSON object."""

    def describe_last(self) -> str:
        """Who answered the last call and how fast, for the chat panel ("cloud 1.2 s")."""
        return ""


class VLMBackend(ABC):
    name = "vlm"

    @abstractmethod
    def ask(self, image_jpeg: bytes, prompt: str, schema: dict | None = None, source: str = "") -> dict:
        """Question about one image, answered as JSON matching `schema`. `source` must be HEAD_CAMERA."""

    def describe_last(self) -> str:
        return ""
