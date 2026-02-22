from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from ash.config import AshConfig
from ash.config.models import ModelConfig
from ash.images.service import ImageUnderstandingService
from ash.images.types import ImageAnalyzeResult
from ash.integrations.image import ImageIntegration
from ash.integrations.runtime import IntegrationContext
from ash.providers.base import ImageAttachment, IncomingMessage


class _FakeImageProvider:
    name = "openai"

    async def analyze(self, request):  # noqa: ANN001
        _ = request
        return ImageAnalyzeResult(
            summary="A whiteboard with a handwritten task list.",
            salient_text="Laundry, Closet, Bathrooms",
            uncertainty="low",
            safety_notes=None,
            provider="openai",
            model="gpt-5.2",
        )


class _FailingImageProvider:
    name = "openai"

    async def analyze(self, request):  # noqa: ANN001
        _ = request
        raise RuntimeError("boom")


def _config() -> AshConfig:
    return AshConfig(
        workspace=Path("tmp-workspace"),
        models={"default": ModelConfig(provider="openai", model="gpt-5-mini")},
    )


def _message(*, text: str) -> IncomingMessage:
    return IncomingMessage(
        id="m-1",
        chat_id="c-1",
        user_id="u-1",
        text=text,
        images=[
            ImageAttachment(
                file_id="file-1",
                mime_type="image/png",
                data=b"fake-image",
                width=800,
                height=600,
            )
        ],
    )


@pytest.mark.asyncio
async def test_image_service_injects_context_block() -> None:
    service = ImageUnderstandingService(config=_config(), provider=_FakeImageProvider())
    message = _message(text="what's on this board?")

    updated = await service.preprocess_message(message)

    assert "[IMAGE_CONTEXT]" in updated.text
    assert "summary: A whiteboard with a handwritten task list." in updated.text
    assert "what's on this board?" in updated.text
    assert updated.metadata["image.processed"] is True


@pytest.mark.asyncio
async def test_image_service_handles_no_caption_with_followup_instruction() -> None:
    service = ImageUnderstandingService(config=_config(), provider=_FakeImageProvider())
    message = _message(text="")

    updated = await service.preprocess_message(message)

    assert "clarifying follow-up question" in updated.text
    assert "[IMAGE_CONTEXT]" in updated.text


@pytest.mark.asyncio
async def test_image_integration_falls_back_on_provider_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        "ash.integrations.image.OpenAIImageProvider",
        lambda: _FailingImageProvider(),
    )
    config = _config()
    context = IntegrationContext(
        config=config,
        components=cast(Any, SimpleNamespace()),
        mode="serve",
    )
    integration = ImageIntegration()
    await integration.setup(context)

    message = _message(text="")
    updated = await integration.preprocess_incoming_message(message, context)

    assert "Ask one clarifying question" in updated.text
