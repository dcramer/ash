"""Image understanding service orchestration."""

from __future__ import annotations

import logging

from ash.config import AshConfig
from ash.images.base import ImageProvider
from ash.images.types import ImageAnalyzeRequest, ImageAnalyzeResult, ImageInput
from ash.providers.base import ImageAttachment, IncomingMessage

logger = logging.getLogger("image")


def _resolve_image_model(config: AshConfig) -> str:
    configured = (config.image.model or "").strip()
    if configured:
        if "/" in configured:
            provider, model = configured.split("/", 1)
            if provider.lower() == "openai" and model.strip():
                return model.strip()
        if configured in config.models:
            alias_model = config.get_model(configured)
            if alias_model.provider == "openai":
                return alias_model.model
        return configured

    default_model = config.default_model
    if default_model.provider == "openai":
        return default_model.model
    return "gpt-5.2"


class ImageUnderstandingService:
    """Coordinates provider call and message text injection."""

    def __init__(self, *, config: AshConfig, provider: ImageProvider) -> None:
        self._config = config
        self._provider = provider
        self._model = _resolve_image_model(config)

    def _select_images(self, message: IncomingMessage) -> list[ImageInput]:
        selected: list[ImageInput] = []
        max_images = self._config.image.max_images_per_message
        max_bytes = self._config.image.max_image_bytes

        for attachment in message.images[:max_images]:
            prepared = self._prepare_attachment(attachment, max_bytes=max_bytes)
            if prepared is not None:
                selected.append(prepared)
        return selected

    @staticmethod
    def _prepare_attachment(
        attachment: ImageAttachment,
        *,
        max_bytes: int,
    ) -> ImageInput | None:
        if attachment.data is None:
            return None
        if len(attachment.data) > max_bytes:
            return None
        mime_type = (
            attachment.mime_type or "image/jpeg"
        ).lower().strip() or "image/jpeg"
        if not mime_type.startswith("image/"):
            return None
        return ImageInput(
            data=attachment.data,
            mime_type=mime_type,
            width=attachment.width,
            height=attachment.height,
        )

    def _build_prompt(self, message: IncomingMessage) -> str:
        include_ocr = (
            "Include visible text in salient_text."
            if self._config.image.include_ocr_text
            else "If visible text is present, you may summarize it."
        )
        user_text = message.text.strip()
        if user_text:
            return (
                "Analyze the image(s) for this user request.\n"
                f"User request: {user_text}\n"
                f"{include_ocr}"
            )
        return (
            "Analyze the image(s) and describe what is most relevant for a helpful assistant reply.\n"
            f"{include_ocr}"
        )

    @staticmethod
    def _format_context_block(result: ImageAnalyzeResult) -> str:
        lines = [
            "[IMAGE_CONTEXT]",
            f"- summary: {result.summary}",
            f"- salient_text: {result.salient_text}",
            f"- uncertainty: {result.uncertainty}",
        ]
        if result.safety_notes:
            lines.append(f"- safety_notes: {result.safety_notes}")
        lines.append("[/IMAGE_CONTEXT]")
        return "\n".join(lines)

    def _inject_context(self, message: IncomingMessage, block: str) -> IncomingMessage:
        original = message.text.strip()
        if not original:
            if self._config.image.no_caption_auto_respond:
                original = (
                    "The user sent image(s) without caption. "
                    "Respond with a concise description and one clarifying follow-up question."
                )
            else:
                original = "The user sent image(s) without caption."

        if self._config.image.inject_position == "append":
            message.text = f"{original}\n\n{block}"
        else:
            message.text = f"{block}\n\n{original}"
        return message

    async def preprocess_message(self, message: IncomingMessage) -> IncomingMessage:
        if not message.has_images:
            return message
        images = self._select_images(message)
        if not images:
            return message

        request = ImageAnalyzeRequest(
            prompt=self._build_prompt(message),
            model=self._model,
            images=images,
            timeout_seconds=self._config.image.request_timeout_seconds,
        )
        result = await self._provider.analyze(request)

        message.metadata["image.processed"] = True
        message.metadata["image.provider"] = result.provider
        if result.model:
            message.metadata["image.model"] = result.model
        return self._inject_context(message, self._format_context_block(result))
