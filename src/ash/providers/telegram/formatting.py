"""Telegram text formatting helpers."""

from aiogram.enums import ParseMode

_MARKDOWN_V2_SPECIAL_CHARS = r"_*[]()~`>#+-=|{}.!"


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    return "".join(
        f"\\{char}" if char in _MARKDOWN_V2_SPECIAL_CHARS else char for char in text
    )


def render_text_for_parse_mode(text: str, parse_mode: ParseMode | None) -> str:
    """Render text for the configured Telegram parse mode."""
    if parse_mode == ParseMode.MARKDOWN_V2:
        return escape_markdown_v2(text)
    return text


def rendered_text_length(text: str, parse_mode: ParseMode | None) -> int:
    """Return payload length after parse-mode rendering."""
    return len(render_text_for_parse_mode(text, parse_mode))


def truncate_for_rendered_limit(
    text: str,
    parse_mode: ParseMode | None,
    max_length: int,
) -> str:
    """Return the longest raw-text prefix whose rendered payload fits max_length."""
    if max_length <= 0 or not text:
        return ""
    if rendered_text_length(text, parse_mode) <= max_length:
        return text

    lo = 0
    hi = len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if rendered_text_length(text[:mid], parse_mode) <= max_length:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]
