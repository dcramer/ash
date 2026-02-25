"""Browser tool exposing structured browser actions."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from ash.browser import BrowserManager, format_browser_result
from ash.tools.base import Tool, ToolContext, ToolResult


class BrowserTool(Tool):
    """Structured browser action tool.

    Actions are deterministic and session-centric.
    """

    def __init__(self, manager: BrowserManager):
        self._manager = manager

    @property
    def name(self) -> str:
        return "browser"

    @property
    def description(self) -> str:
        return (
            "Perform deterministic browser actions using named sessions. "
            "Use for interactive/dynamic/authenticated pages (click/type/wait/screenshot), "
            "not simple web lookups that `web_search`/`web_fetch` can handle."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        provider_enum = list(self._manager.provider_names)
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": [
                        "session.start",
                        "session.list",
                        "session.show",
                        "session.close",
                        "session.archive",
                        "page.goto",
                        "page.extract",
                        "page.click",
                        "page.type",
                        "page.wait_for",
                        "page.screenshot",
                    ],
                    "description": "Browser action to execute.",
                },
                "provider": {
                    "type": "string",
                    "enum": provider_enum,
                    "description": "Optional provider override.",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID to target.",
                },
                "session_name": {
                    "type": "string",
                    "description": "Session name to target/create.",
                },
                "profile_name": {
                    "type": "string",
                    "description": "Optional named profile.",
                },
                "url": {
                    "type": "string",
                    "description": "Target URL for page.goto.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["text", "title"],
                    "description": "Extract mode for page.extract.",
                },
                "selector": {
                    "type": "string",
                    "description": "DOM selector for click/type/wait/extract.",
                },
                "text": {
                    "type": "string",
                    "description": "Text for page.type.",
                },
                "clear_first": {
                    "type": "boolean",
                    "description": "Clear field before typing.",
                    "default": True,
                },
                "timeout_seconds": {
                    "type": "number",
                    "description": "Optional action timeout.",
                },
                "max_chars": {
                    "type": "integer",
                    "description": "Max characters for extracted output.",
                    "default": 3000,
                },
            },
            "required": ["action"],
        }

    async def execute(
        self, input_data: dict[str, Any], context: ToolContext
    ) -> ToolResult:
        action = str(input_data.get("action") or "").strip()
        if not action:
            return ToolResult.error("missing required field: action")
        effective_user_id = (context.user_id or "").strip()
        if not effective_user_id:
            return ToolResult.error("browser requires authenticated user context")

        result = await self._manager.execute_action(
            action=action,
            effective_user_id=effective_user_id,
            provider_name=(
                str(input_data["provider"]) if input_data.get("provider") else None
            ),
            session_id=(
                str(input_data["session_id"]) if input_data.get("session_id") else None
            ),
            session_name=(
                str(input_data["session_name"])
                if input_data.get("session_name")
                else None
            ),
            profile_name=(
                str(input_data["profile_name"])
                if input_data.get("profile_name")
                else None
            ),
            params=dict(input_data),
        )
        payload = format_browser_result(result)
        if result.ok:
            metadata: dict[str, Any] = {}
            if result.page_url:
                metadata["page_url"] = result.page_url
                host = (urlparse(result.page_url).netloc or "").strip().lower()
                if host.startswith("www."):
                    host = host[4:]
                if host:
                    metadata["domain"] = host
            return ToolResult.success(payload, **metadata)
        return ToolResult.error(payload, error_code=result.error_code)
