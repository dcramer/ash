"""Telegram checkpoint UI helpers for inline keyboard interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardMarkup

# Callback data prefix for checkpoint interactions
CALLBACK_PREFIX = "chkpt"

# Maximum length for Telegram callback data (64 bytes)
MAX_CALLBACK_DATA_LEN = 64


def create_checkpoint_keyboard(checkpoint: dict[str, Any]) -> InlineKeyboardMarkup:
    """Create an inline keyboard from checkpoint options.

    Args:
        checkpoint: Checkpoint dict with 'checkpoint_id' and optional 'options'.

    Returns:
        InlineKeyboardMarkup with buttons for each option.
    """
    from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

    checkpoint_id = checkpoint.get("checkpoint_id", "")
    options = checkpoint.get("options") or ["Proceed", "Cancel"]

    # Truncate checkpoint_id to fit in callback data
    # Format: "chkpt:{id}:{index}" - allow ~40 chars for ID
    max_id_len = MAX_CALLBACK_DATA_LEN - len(CALLBACK_PREFIX) - 5  # ":" + ":" + "XX"
    truncated_id = checkpoint_id[:max_id_len]

    buttons = []
    for idx, option in enumerate(options):
        callback_data = f"{CALLBACK_PREFIX}:{truncated_id}:{idx}"
        buttons.append([InlineKeyboardButton(text=option, callback_data=callback_data)])

    return InlineKeyboardMarkup(inline_keyboard=buttons)


def parse_callback_data(data: str) -> tuple[str, int] | None:
    """Parse callback data to extract checkpoint info.

    Args:
        data: Callback data string in format "chkpt:{checkpoint_id}:{option_index}".

    Returns:
        Tuple of (checkpoint_id, option_index) or None if invalid format.
    """
    if not data.startswith(f"{CALLBACK_PREFIX}:"):
        return None

    parts = data.split(":", 2)
    if len(parts) != 3:
        return None

    _, checkpoint_id, index_str = parts
    try:
        option_index = int(index_str)
    except ValueError:
        return None

    return checkpoint_id, option_index


def format_checkpoint_message(checkpoint: dict[str, Any]) -> str:
    """Format checkpoint prompt for display.

    Args:
        checkpoint: Checkpoint dict with 'prompt' and optional 'options'.

    Returns:
        Formatted message string.
    """
    prompt = checkpoint.get("prompt", "Agent paused for input")
    # Options are shown as clickable buttons, no need to include in text
    return f"**Agent paused for input**\n\n{prompt}"
