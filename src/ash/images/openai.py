"""OpenAI-backed image understanding provider."""

from __future__ import annotations

import base64
import json
from typing import Any, Literal, cast

import openai

from ash.images.base import ImageProvider
from ash.images.types import (
    ImageAnalyzeRequest,
    ImageAnalyzeResult,
)

_SYSTEM_PROMPT = """You analyze user-provided images for an assistant.
Return JSON with keys:
- summary: concise factual description (1-3 sentences)
- salient_text: notable text visible in the image, or "none"
- uncertainty: one of low|medium|high
- safety_notes: optional short note, else null
Do not include markdown fences or extra keys.
"""


def _parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("openai_image_invalid_json")


def _extract_output_text(response: Any) -> str:
    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        if getattr(item, "type", None) != "message":
            continue
        for part in getattr(item, "content", []) or []:
            if getattr(part, "type", None) == "output_text":
                parts.append(part.text)
    return "\n".join(parts).strip()


class OpenAIImageProvider(ImageProvider):
    """OpenAI Responses API image understanding implementation."""

    def __init__(self, api_key: str | None = None) -> None:
        self._client = openai.AsyncOpenAI(api_key=api_key)

    @property
    def name(self) -> str:
        return "openai"

    async def analyze(self, request: ImageAnalyzeRequest) -> ImageAnalyzeResult:
        input_content: list[dict[str, Any]] = [
            {"type": "input_text", "text": request.prompt}
        ]
        for image in request.images:
            image_data = base64.b64encode(image.data).decode("ascii")
            input_content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:{image.mime_type};base64,{image_data}",
                }
            )

        kwargs: dict[str, Any] = {
            "model": request.model,
            "input": [{"role": "user", "content": input_content}],
            "instructions": _SYSTEM_PROMPT,
            "max_output_tokens": 500,
            "timeout": request.timeout_seconds,
        }
        response = await self._client.responses.create(**kwargs)

        text = _extract_output_text(response)
        payload = _parse_json_object(text)
        uncertainty = str(payload.get("uncertainty", "medium")).lower()
        if uncertainty not in {"low", "medium", "high"}:
            uncertainty = "medium"
        summary = str(payload.get("summary", "")).strip()
        salient_text = str(payload.get("salient_text", "none")).strip() or "none"
        safety_notes_raw = payload.get("safety_notes")
        safety_notes = str(safety_notes_raw).strip() if safety_notes_raw else None
        if not summary:
            raise ValueError("openai_image_missing_summary")

        return ImageAnalyzeResult(
            summary=summary,
            salient_text=salient_text,
            uncertainty=cast(Literal["low", "medium", "high"], uncertainty),
            safety_notes=safety_notes,
            provider=self.name,
            model=getattr(response, "model", request.model),
            raw=response.model_dump(),
        )
