"""Telegram message handling utilities.

Formatting, escaping, and helper functions for Telegram message handling.
"""

from typing import TYPE_CHECKING, Any

from ash.providers.telegram.formatting import escape_markdown_v2 as _escape_markdown_v2

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

# Constants
MAX_MESSAGE_LENGTH = 4096  # Telegram message limit
STREAM_DELAY = 5.0  # Start showing partial response after this many seconds
MIN_EDIT_INTERVAL = 1.0  # Minimum time between edits


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format."""
    return _escape_markdown_v2(text)


def truncate_str(s: str, max_len: int) -> str:
    """Truncate string (first line only, max length)."""
    first_line, *rest = s.split("\n", 1)
    truncated = len(first_line) > max_len or bool(rest)
    return first_line[:max_len] + "..." if truncated else first_line


def get_filename(path: str) -> str:
    """Extract filename from a path."""
    return path.rsplit("/", 1)[-1] if "/" in path else path


def get_domain(url: str) -> str:
    """Extract domain from a URL."""
    if "://" in url:
        return url.split("://", 1)[1].split("/")[0]
    return url.split("/")[0]


def resolve_agent_model(
    agent_name: str,
    config: "AshConfig | None",
    agent_registry: "AgentRegistry | None",
) -> str | None:
    """Resolve the model name for an agent, considering config overrides."""
    if not (agent_registry and config and agent_name in agent_registry):
        return None
    agent = agent_registry.get(agent_name)
    override = config.agents.get(agent_name)
    return override.model if override and override.model else agent.config.model


def resolve_skill_model(
    skill_name: str,
    config: "AshConfig | None",
    skill_registry: "SkillRegistry | None",
) -> str | None:
    """Resolve the model name for a skill, considering config overrides."""
    if not (skill_registry and config and skill_registry.has(skill_name)):
        return None
    skill = skill_registry.get(skill_name)
    skill_config = config.skills.get(skill_name)
    return skill_config.model if skill_config and skill_config.model else skill.model


def format_tool_brief(
    tool_name: str,
    tool_input: dict[str, Any],
    config: "AshConfig | None" = None,
    agent_registry: "AgentRegistry | None" = None,
    skill_registry: "SkillRegistry | None" = None,
) -> str:
    """Format tool execution into a brief status message."""
    match tool_name:
        case "bash":
            return f"Running: `{truncate_str(tool_input.get('command', ''), 50)}`"
        case "web_search":
            return f"Searching: {truncate_str(tool_input.get('query', ''), 40)}"
        case "web_fetch":
            return f"Reading: {get_domain(tool_input.get('url', ''))}"
        case "use_agent":
            agent_name = tool_input.get("agent", "unknown")
            model = resolve_agent_model(agent_name, config, agent_registry)
            suffix = f" ({model})" if model else ""
            preview = truncate_str(tool_input.get("message", ""), 150)
            return f"{agent_name}{suffix}: {preview}"
        case "write_file":
            return f"Writing: {get_filename(tool_input.get('file_path', ''))}"
        case "read_file":
            return f"Reading: {get_filename(tool_input.get('file_path', ''))}"
        case "remember":
            return "Saving to memory"
        case "recall":
            query = truncate_str(tool_input.get("query", ""), 30)
            return f"Searching memories: {query}" if query else "Searching memories"
        case "use_skill":
            skill_name = tool_input.get("skill", "unknown")
            model = resolve_skill_model(skill_name, config, skill_registry)
            suffix = f" ({model})" if model else ""
            preview = truncate_str(tool_input.get("message", ""), 150)
            return f"{skill_name}{suffix}: {preview}"
        case _:
            display_name = tool_name.replace("_tool", "").replace("_", " ")
            return f"Running: {display_name}"


def extract_text_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from content blocks."""
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else ""


def merge_progress_and_response(
    progress_messages: list[str], response_text: str
) -> str:
    """Build final content while avoiding duplicate trailing response text."""
    if not progress_messages:
        return response_text
    response = response_text.strip()
    if response and progress_messages[-1].strip() == response:
        return "\n".join(progress_messages)
    parts = progress_messages + (["", response_text] if response_text else [])
    return "\n".join(parts)


def append_inline_attribution(response_text: str, attribution: str | None) -> str:
    """Append provenance attribution as a concise inline sentence."""
    if not attribution:
        return response_text
    text = response_text.strip()
    if not text:
        return attribution
    if attribution in text:
        return response_text
    if text.endswith((".", "!", "?")):
        return f"{text} {attribution}"
    return f"{text}. {attribution}"
