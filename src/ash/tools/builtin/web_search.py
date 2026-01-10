"""Web search tool using Brave Search API, executed in sandbox."""

import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.sandbox import SandboxExecutor
from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig
from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.config.models import SandboxConfig

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Python script to execute inside sandbox
# This is more robust than curl+jq for URL encoding and JSON parsing
SEARCH_SCRIPT = '''
import json, os, sys, urllib.request, urllib.parse

query = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 5

api_key = os.environ.get("BRAVE_API_KEY", "")
if not api_key:
    print("ERROR: BRAVE_API_KEY not set", file=sys.stderr)
    sys.exit(1)

q = urllib.parse.quote(query)
url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={count}"

try:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "X-Subscription-Token": api_key,
        }
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        if resp.status != 200:
            print(f"ERROR: HTTP {resp.status}", file=sys.stderr)
            sys.exit(1)
        data = json.load(resp)
except urllib.error.HTTPError as e:
    if e.code == 401:
        print("ERROR: Invalid API key", file=sys.stderr)
    elif e.code == 429:
        print("ERROR: Rate limit exceeded", file=sys.stderr)
    else:
        print(f"ERROR: HTTP {e.code}", file=sys.stderr)
    sys.exit(1)
except urllib.error.URLError as e:
    print(f"ERROR: {e.reason}", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {e}", file=sys.stderr)
    sys.exit(1)

results = data.get("web", {}).get("results", [])
if not results:
    print("No results found")
    sys.exit(0)

for i, r in enumerate(results, 1):
    title = r.get("title", "No title")
    url = r.get("url", "")
    desc = r.get("description", "")
    # Truncate long descriptions
    if len(desc) > 300:
        desc = desc[:297] + "..."
    print(f"{i}. {title}")
    print(f"   URL: {url}")
    print(f"   {desc}")
    print()
'''


class WebSearchTool(Tool):
    """Search the web using Brave Search API.

    All requests execute inside the Docker sandbox for network control.
    Requires network_mode: bridge in sandbox configuration.
    """

    def __init__(
        self,
        api_key: str,
        sandbox_config: "SandboxConfig | None" = None,
        workspace_path: Path | None = None,
        max_results: int = 10,
    ):
        """Initialize web search tool.

        Args:
            api_key: Brave Search API key.
            sandbox_config: Sandbox configuration (pydantic model from config).
            workspace_path: Path to workspace (for sandbox config).
            max_results: Maximum results to return per search.
        """
        self._api_key = api_key
        self._max_results = max_results
        self._sandbox_config = sandbox_config

        # Check network mode
        network_mode = sandbox_config.network_mode if sandbox_config else "bridge"
        if network_mode == "none":
            raise ValueError(
                "Web search requires network_mode: bridge in sandbox configuration"
            )

        # Build sandbox config with API key in environment
        manager_config = self._build_manager_config(sandbox_config, workspace_path)
        self._executor = SandboxExecutor(
            config=manager_config,
            environment={"BRAVE_API_KEY": api_key},
        )

    def _build_manager_config(
        self,
        config: "SandboxConfig | None",
        workspace_path: Path | None,
    ) -> SandboxManagerConfig:
        """Convert pydantic SandboxConfig to manager's dataclass config."""
        if config is None:
            # Default to bridge mode for web search
            return SandboxManagerConfig(
                workspace_path=workspace_path,
                network_mode="bridge",
            )

        return SandboxManagerConfig(
            image=config.image,
            timeout=config.timeout,
            memory_limit=config.memory_limit,
            cpu_limit=config.cpu_limit,
            runtime=config.runtime,
            network_mode=config.network_mode,
            dns_servers=list(config.dns_servers) if config.dns_servers else [],
            http_proxy=config.http_proxy,
            workspace_path=workspace_path,
            workspace_access=config.workspace_access,
        )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return (
            "Search the web for current information. "
            "Use this to find recent news, documentation, articles, or any "
            "information that may not be in your training data."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query.",
                },
                "count": {
                    "type": "integer",
                    "description": f"Number of results (max {self._max_results}).",
                    "default": 5,
                },
            },
            "required": ["query"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute web search in sandbox.

        Args:
            input_data: Must contain 'query' key.
            context: Execution context.

        Returns:
            Tool result with search results.
        """
        query = input_data.get("query", "").strip()
        if not query:
            return ToolResult.error("Missing required parameter: query")

        count = min(input_data.get("count", 5), self._max_results)

        try:
            # Build command to execute Python search script
            # Query is passed as argument, properly escaped
            escaped_query = shlex.quote(query)
            command = f"python3 -c {shlex.quote(SEARCH_SCRIPT)} {escaped_query} {count}"

            result = await self._executor.execute(
                command,
                timeout=30,
                reuse_container=True,
            )

            if result.timed_out:
                return ToolResult.error("Search request timed out")

            # Check for errors in stderr
            if result.stderr:
                stderr = result.stderr.strip()
                if stderr.startswith("ERROR:"):
                    error_msg = stderr.replace("ERROR:", "").strip()
                    return ToolResult.error(f"Search failed: {error_msg}")

            # Return results
            output = result.stdout.strip() if result.stdout else ""
            if not output or output == "No results found":
                return ToolResult.success(
                    f"No results found for: {query}",
                    result_count=0,
                )

            # Count results (each result starts with a number followed by dot)
            result_count = sum(
                1 for line in output.split("\n") if line and line[0].isdigit() and ". " in line
            )

            return ToolResult.success(
                output,
                result_count=result_count,
            )

        except Exception as e:
            return ToolResult.error(f"Search error: {e}")

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._executor:
            await self._executor.cleanup()
