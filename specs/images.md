# Images

Inbound image understanding is implemented as an integration contributor.

Files:
- `src/ash/integrations/image.py`
- `src/ash/images/service.py`
- `src/ash/images/openai.py`
- `src/ash/providers/base.py` (`IncomingMessage.images`)

## Architecture

- Telegram provider downloads image bytes into `IncomingMessage.images[*].data`.
- `ImageIntegration.preprocess_incoming_message` runs before normal session/agent flow.
- Image analysis is delegated to `ImageUnderstandingService`, which uses an image provider (`OpenAIImageProvider` in phase 1).
- Service injects structured `[IMAGE_CONTEXT]` text into `message.text`.
- Agent pipeline remains text-based; no multimodal session schema changes in phase 1.

## Configuration

```toml
[image]
enabled = true
provider = "openai"
model = "gpt-5.2"            # alias or provider/model; optional
max_images_per_message = 1
max_image_bytes = 8000000
request_timeout_seconds = 12.0
include_ocr_text = true
inject_position = "prepend"  # prepend|append
no_caption_auto_respond = true
```

## Behavior

- If a message has images and valid image bytes, image understanding runs.
- Injected block format:

```text
[IMAGE_CONTEXT]
- summary: ...
- salient_text: ...
- uncertainty: low|medium|high
- safety_notes: ...
[/IMAGE_CONTEXT]
```

- If user did not include text and `no_caption_auto_respond=true`, injected text asks the agent to provide a concise description and one clarifying follow-up question.
- On provider failure, processing falls back safely to normal text flow (with a no-caption clarifying fallback for empty image-only messages).

## Logging

- `image_preprocess_started`
- `image_preprocess_succeeded`
- `image_preprocess_failed`

No raw image bytes or base64 payloads are logged.
