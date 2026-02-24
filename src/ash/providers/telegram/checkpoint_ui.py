"""Telegram checkpoint UI helpers for inline keyboard interactions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from aiogram.types import InlineKeyboardMarkup

# Callback data prefix for checkpoint interactions
CALLBACK_PREFIX = "chkpt"

# Maximum length for Telegram callback data (64 bytes)
MAX_CALLBACK_DATA_LEN = 64

# Maximum length for truncated checkpoint IDs
# Format: "chkpt:{id}:{index}" — reserve space for prefix, colons, and 2-digit index
MAX_CHECKPOINT_ID_LEN = MAX_CALLBACK_DATA_LEN - len(CALLBACK_PREFIX) - 5


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

    truncated_id = checkpoint_id[:MAX_CHECKPOINT_ID_LEN]

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
    prompt = str(checkpoint.get("prompt", "Agent paused for input"))
    options = checkpoint.get("options") or []
    if not options:
        return prompt
    choices = ", ".join(str(option) for option in options)
    return f"{prompt}\n\nSuggested responses: {choices}"
