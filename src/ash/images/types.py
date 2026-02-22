"""Types for inbound image understanding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(slots=True)
class ImageInput:
    """A single image input for understanding."""

    data: bytes
    mime_type: str
    width: int | None = None
    height: int | None = None


@dataclass(slots=True)
class ImageAnalyzeRequest:
    """Request to analyze one or more inbound images."""

    prompt: str
    model: str
    images: list[ImageInput]
    timeout_seconds: float = 12.0


@dataclass(slots=True)
class ImageAnalyzeResult:
    """Structured output from an image analysis provider."""

    summary: str
    salient_text: str
    uncertainty: Literal["low", "medium", "high"]
    safety_notes: str | None = None
    provider: str = "unknown"
    model: str | None = None
    raw: dict[str, Any] | None = None
