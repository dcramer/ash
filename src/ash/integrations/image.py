"""Image understanding integration contributor.

Spec contract: specs/subsystems.md (Integration Hooks).
"""

from __future__ import annotations

import logging
import os
import time
from typing import TYPE_CHECKING

from ash.images import ImageUnderstandingService, OpenAIImageProvider
from ash.integrations.runtime import IntegrationContext, IntegrationContributor

if TYPE_CHECKING:
    from ash.providers.base import IncomingMessage

logger = logging.getLogger("image")


class ImageIntegration(IntegrationContributor):
    """Preprocesses inbound image messages into structured text context."""

    name = "image"
    priority = 150

    def __init__(self) -> None:
        self._service: ImageUnderstandingService | None = None

    async def setup(self, context: IntegrationContext) -> None:
        if not context.config.image.enabled:
            return
        if context.config.image.provider != "openai":
            logger.warning(
                "image_preprocess_skipped",
                extra={"skip_reason": "unsupported_provider"},
            )
            return
        api_key = context.config.resolve_provider_api_key("openai")
        resolved_api_key = api_key.get_secret_value() if api_key else None
        if not resolved_api_key:
            resolved_api_key = os.environ.get("OPENAI_API_KEY")
        if not resolved_api_key:
            logger.warning(
                "image_preprocess_skipped",
                extra={"skip_reason": "missing_openai_api_key"},
            )
            return
        try:
            self._service = ImageUnderstandingService(
                config=context.config,
                provider=OpenAIImageProvider(api_key=resolved_api_key),
            )
        except ValueError as e:
            logger.warning(
                "image_preprocess_skipped",
                extra={
                    "skip_reason": "invalid_image_model_config",
                    "error.message": str(e),
                },
            )
            self._service = None

    async def preprocess_incoming_message(
        self,
        message: IncomingMessage,
        context: IntegrationContext,
    ) -> IncomingMessage:
        if self._service is None or not message.has_images:
            return message

        started = time.monotonic()
        logger.info(
            "image_preprocess_started",
            extra={"image.count": len(message.images)},
        )
        try:
            updated = await self._service.preprocess_message(message)
        except Exception as e:
            duration_ms = int((time.monotonic() - started) * 1000)
            logger.warning(
                "image_preprocess_failed",
                extra={"error.message": str(e), "duration_ms": duration_ms},
            )
            # Keep flow safe and deterministic: no image context on failure.
            if (
                not message.text.strip()
                and context.config.image.no_caption_auto_respond
            ):
                message.text = (
                    "The user sent image(s) without caption. "
                    "Ask one clarifying question about what they want to know."
                )
            return message

        duration_ms = int((time.monotonic() - started) * 1000)
        if not bool(updated.metadata.get("image.processed")):
            logger.warning(
                "image_preprocess_skipped",
                extra={
                    "skip_reason": str(
                        updated.metadata.get("image.skip_reason") or "unknown"
                    ),
                    "image.count": len(message.images),
                    "duration_ms": duration_ms,
                },
            )
            return updated

        logger.info(
            "image_preprocess_succeeded",
            extra={
                "image.count": len(message.images),
                "duration_ms": duration_ms,
            },
        )
        return updated
