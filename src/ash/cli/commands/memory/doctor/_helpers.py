"""Shared helpers for doctor subcommands."""

from __future__ import annotations

import json
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any

import typer

from ash.cli.console import dim

if TYPE_CHECKING:
    from ash.config.models import AshConfig
    from ash.llm.base import LLMProvider


def normalize_for_comparison(s: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def is_trivial_rewrite(old: str, new: str, threshold: float = 0.92) -> bool:
    """Check if a proposed rewrite is trivially similar to the original."""
    norm_old = normalize_for_comparison(old)
    norm_new = normalize_for_comparison(new)
    return SequenceMatcher(None, norm_old, norm_new).ratio() > threshold


# Pattern matching [PLACEHOLDER] tokens like [Name], [DATE], [LOCATION]
_PLACEHOLDER_RE = re.compile(r"\[[A-Z][A-Za-z_\s]*\]")


def has_placeholder(text: str) -> bool:
    """Check if text contains bracket placeholders like [Name], [DATE], etc."""
    return bool(_PLACEHOLDER_RE.search(text))


# Life event keywords that should never be archived
_LIFE_EVENT_KEYWORDS = (
    "pregnant",
    "expecting",
    "baby",
    "due date",
    "wedding",
    "marriage",
    "married",
    "divorce",
    "divorced",
    "moving",
    "relocated",
    "graduation",
    "graduated",
)

# Date-like patterns: "August 19, 2026", "2026-08-19", "March 15", "08/19/2026"
_DATE_RE = re.compile(
    r"\b(?:"
    r"\d{4}[-/]\d{1,2}[-/]\d{1,2}"
    r"|"
    r"\d{1,2}[-/]\d{1,2}[-/]\d{4}"
    r"|"
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}"
    r")\b",
    re.IGNORECASE,
)


def should_block_archive(content: str, reason: str) -> str | None:
    """Check if an archive decision should be blocked.

    Returns a human-readable reason string if blocked, None if OK to archive.
    """
    lower = content.lower()

    matched = next((kw for kw in _LIFE_EVENT_KEYWORDS if kw in lower), None)
    if matched:
        return f"life event keyword '{matched}'"

    if reason == "negative_knowledge" and _DATE_RE.search(content):
        return "contains specific date"

    return None


def truncate(text: str, length: int = 120) -> str:
    """Truncate text to length, replacing newlines with spaces."""
    flat = text.replace("\n", " ")
    if len(flat) > length:
        return flat[:length] + "..."
    return flat


def create_llm(config: AshConfig) -> tuple[LLMProvider, str]:
    """Create an LLM provider from config. Returns (provider, model_name)."""
    from ash.llm import create_llm_provider

    model_config = config.default_model
    api_key = config.resolve_api_key("default")
    llm = create_llm_provider(
        model_config.provider,
        api_key=api_key.get_secret_value() if api_key else None,
    )
    return llm, model_config.model


async def llm_complete(
    llm: LLMProvider, model: str, prompt: str, max_tokens: int = 1024
) -> dict[str, Any]:
    """Send a prompt to the LLM and parse the JSON response."""
    from ash.llm.types import Message, Role

    response = await llm.complete(
        messages=[Message(role=Role.USER, content=prompt)],
        model=model,
        max_tokens=max_tokens,
        temperature=0.1,
    )
    return parse_json_from_response(response.message.get_text())


def confirm_or_cancel(prompt: str, force: bool) -> bool:
    """Return True if the user confirms (or force is set). Print cancel on decline."""
    if force:
        return True
    confirmed = typer.confirm(prompt)
    if not confirmed:
        dim("Cancelled")
    return confirmed


def parse_json_from_response(text: str) -> dict[str, Any]:
    """Extract JSON from an LLM response, handling markdown code blocks and trailing text."""
    text = text.strip()

    # Strip markdown code fences if present
    match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        text = match.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # raw_decode stops at the end of the first valid JSON object,
        # handling trailing commentary from the LLM
        result, _ = json.JSONDecoder().raw_decode(text)
        if not isinstance(result, dict):
            raise
        return result
