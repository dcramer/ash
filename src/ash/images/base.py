"""Image understanding provider interface."""

from __future__ import annotations

from typing import Protocol

from ash.images.types import ImageAnalyzeRequest, ImageAnalyzeResult


class ImageProvider(Protocol):
    """Provider contract for image analysis."""

    @property
    def name(self) -> str:
        """Stable provider name."""
        ...

    async def analyze(self, request: ImageAnalyzeRequest) -> ImageAnalyzeResult:
        """Analyze images and return structured description."""
        ...
