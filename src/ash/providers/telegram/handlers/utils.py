"""Telegram message handling utilities.

Formatting, escaping, and helper functions for Telegram message handling.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ash.agents import AgentRegistry
    from ash.config import AshConfig
    from ash.skills import SkillRegistry

# Constants
MAX_MESSAGE_LENGTH = 4096  # Telegram message limit
STREAM_DELAY = 5.0  # Start showing partial response after this many seconds
MIN_EDIT_INTERVAL = 1.0  # Minimum time between edits


def escape_markdown_v2(text: str) -> str:
    """Escape special characters for Telegram MarkdownV2 format.

    Telegram supports two markdown modes: MARKDOWN (legacy) and MARKDOWN_V2.
    MarkdownV2 requires ALL special characters to be escaped with backslash,
    even inside regular text. This is different from standard markdown.

    Special characters that MUST be escaped in MarkdownV2:
        _ * [ ] ( ) ~ ` > # + - = | { } . !

    Example:
        escape_markdown_v2("Hello...") → "Hello\\.\\.\\."
        escape_markdown_v2("(test)") → "\\(test\\)"

    When to use:
        - Always escape user-provided text before including in MarkdownV2 messages
        - Status/thinking messages use MarkdownV2 for consistent formatting
        - Final responses use regular MARKDOWN (more forgiving, less escaping)

    Note:
        In Python string literals, backslashes must be doubled.
        So "_Thinking\\\\.\\\\.\\\\._" becomes "_Thinking\\.\\.\\._" at runtime,
        which Telegram interprets as italic "Thinking...".
    """
    special_chars = r"_*[]()~`>#+-=|{}.!"
    return "".join(f"\\{c}" if c in special_chars else c for c in text)


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


def format_thinking_status(num_tools: int) -> str:
    """Format a thinking status line with tool count, pre-escaped for MarkdownV2.

    Returns a MarkdownV2-formatted italic string. All special characters are
    pre-escaped in the string literal (double backslashes in Python source).

    Examples:
        format_thinking_status(0) → "_Thinking\\.\\.\\._"
            Renders as: _Thinking..._  (italic)

        format_thinking_status(2) → "_Thinking\\.\\.\\. \\(2 tool calls\\)_"
            Renders as: _Thinking... (2 tool calls)_  (italic)

    Note:
        This function returns MarkdownV2-escaped text. It must be sent with
        parse_mode="markdown_v2" to render correctly. Using regular MARKDOWN
        mode will show literal backslashes.
    """
    if num_tools == 0:
        return "_Thinking\\.\\.\\._"
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Thinking\\.\\.\\. \\({num_tools} tool {call_word}\\)_"


def format_tool_summary(num_tools: int, elapsed_seconds: float) -> str:
    """Format a summary of tool calls for regular MARKDOWN (not MarkdownV2).

    Returns a MARKDOWN-formatted italic string. Unlike format_thinking_status(),
    this function does NOT escape for MarkdownV2 because the final response
    is edited with regular MARKDOWN mode (more forgiving of special chars).

    Examples:
        format_tool_summary(3, 5.2) → "_Made 3 tool calls in 5.2s_"
            Renders as: _Made 3 tool calls in 5.2s_  (italic)

    Note:
        This is used in final responses, not thinking messages. The period in
        the elapsed time is NOT escaped because regular MARKDOWN doesn't require it.
    """
    call_word = "call" if num_tools == 1 else "calls"
    return f"_Made {num_tools} tool {call_word} in {elapsed_seconds:.1f}s_"


def extract_text_content(content: list[dict[str, Any]]) -> str:
    """Extract text content from content blocks."""
    texts = [
        block.get("text", "")
        for block in content
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n".join(texts) if texts else ""


# Backward compatibility aliases (internal names used in handlers.py)
_truncate_str = truncate_str
_get_filename = get_filename
_get_domain = get_domain
_resolve_agent_model = resolve_agent_model
_resolve_skill_model = resolve_skill_model
_extract_text_content = extract_text_content
