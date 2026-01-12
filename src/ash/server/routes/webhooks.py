"""Webhook routes for provider integrations."""

import logging
from typing import Any

from fastapi import APIRouter, Request, Response

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/telegram")
async def telegram_webhook(request: Request) -> Response:
    """Handle Telegram webhook updates."""
    telegram_provider = getattr(request.app.state, "telegram_provider", None)
    if not telegram_provider:
        logger.error("Telegram provider not configured")
        return Response(status_code=500)

    try:
        # Parse update data
        update_data: dict[str, Any] = await request.json()

        # Process update
        await telegram_provider.process_webhook_update(update_data)

        return Response(status_code=200)

    except Exception:
        logger.exception("Error processing Telegram webhook")
        return Response(status_code=200)  # Return 200 to prevent retries
