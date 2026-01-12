"""Web search tool using Brave Search API, executed in sandbox."""

import json
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.sandbox import SandboxExecutor
from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.builtin.search_cache import SearchCache
from ash.tools.builtin.search_types import SearchResponse
from ash.tools.retry import RetryConfig, with_retry

if TYPE_CHECKING:
    from ash.config.models import SandboxConfig

logger = logging.getLogger(__name__)

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"

# Python script to execute inside sandbox
# Outputs JSON for reliable parsing and accurate result counting
SEARCH_SCRIPT = '''
import json, os, sys, urllib.request, urllib.parse, time
from urllib.parse import urlparse

query = sys.argv[1]
count = int(sys.argv[2]) if len(sys.argv) > 2 else 5

api_key = os.environ.get("BRAVE_API_KEY", "")
if not api_key:
    print(json.dumps({"error": "BRAVE_API_KEY not set", "code": 500}))
    sys.exit(1)

q = urllib.parse.quote(query)
url = f"https://api.search.brave.com/res/v1/web/search?q={q}&count={count}"

start_time = time.time()

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
            print(json.dumps({"error": f"HTTP {resp.status}", "code": resp.status}))
            sys.exit(1)
        data = json.load(resp)
except urllib.error.HTTPError as e:
    error_msg = {
        401: "Invalid API key",
        429: "Rate limit exceeded",
    }.get(e.code, f"HTTP {e.code}")
    print(json.dumps({"error": error_msg, "code": e.code}))
    sys.exit(1)
except urllib.error.URLError as e:
    print(json.dumps({"error": str(e.reason), "code": 0}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e), "code": 0}))
    sys.exit(1)

search_time_ms = int((time.time() - start_time) * 1000)

def truncate_at_word(text, max_len=300):
    """Truncate at word boundary, not mid-word."""
    if len(text) <= max_len:
        return text
    # Find last space before max_len
    truncated = text[:max_len]
    last_space = truncated.rfind(" ")
    if last_space > max_len * 0.7:  # Only use if space is reasonably close
        truncated = truncated[:last_space]
    return truncated.rstrip() + "..."

def extract_site_name(url_str):
    """Extract readable site name from URL."""
    try:
        parsed = urlparse(url_str)
        domain = parsed.netloc
        # Remove www. prefix
        if domain.startswith("www."):
            domain = domain[4:]
        return domain
    except Exception:
        return None

raw_results = data.get("web", {}).get("results", [])
results = []

for r in raw_results:
    title = r.get("title", "No title")
    result_url = r.get("url", "")
    desc = r.get("description", "")

    # Truncate at word boundary
    if desc:
        desc = truncate_at_word(desc, 300)

    results.append({
        "title": title,
        "url": result_url,
        "description": desc,
        "site_name": extract_site_name(result_url),
        "published_date": r.get("page_age"),  # Brave API field
    })

output = {
    "query": query,
    "results": results,
    "total_count": len(results),
    "search_time_ms": search_time_ms,
}

print(json.dumps(output))
'''


class WebSearchTool(Tool):
    """Search the web using Brave Search API.

    All requests execute inside the Docker sandbox for network control.
    Requires network_mode: bridge in sandbox configuration.

    Features:
    - Structured JSON output with citation metadata
    - In-memory caching with 15-min TTL
    - Retry support for transient errors (via retry.py)
    """

    def __init__(
        self,
        api_key: str,
        sandbox_config: "SandboxConfig | None" = None,
        workspace_path: Path | None = None,
        cache: SearchCache | None = None,
        retry_config: RetryConfig | None = None,
        max_results: int = 10,
    ):
        """Initialize web search tool.

        Args:
            api_key: Brave Search API key.
            sandbox_config: Sandbox configuration (pydantic model from config).
            workspace_path: Path to workspace (for sandbox config).
            cache: Optional search cache for result caching.
            retry_config: Optional retry configuration for transient errors.
            max_results: Maximum results to return per search.
        """
        self._api_key = api_key
        self._max_results = max_results
        self._sandbox_config = sandbox_config
        self._cache = cache
        self._retry_config = retry_config or RetryConfig()

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
            "information that may not be in your training data. "
            "Returns structured results with titles, URLs, and descriptions."
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

        # Check cache first
        if self._cache:
            cached = self._cache.get(query)
            if cached is not None and isinstance(cached, SearchResponse):
                logger.debug(f"Cache hit for query: {query}")
                return ToolResult.success(
                    cached.to_formatted_text(),
                    result_count=len(cached.results),
                    cached=True,
                    search_time_ms=cached.search_time_ms,
                )

        try:
            # Use retry wrapper for transient errors
            response = await with_retry(
                lambda: self._execute_search(query, count),
                config=self._retry_config,
                on_retry=lambda attempt, err, delay: logger.warning(
                    f"Search retry {attempt}/{self._retry_config.max_attempts}: "
                    f"{err}, waiting {delay:.1f}s"
                ),
            )

            # Cache the response
            if self._cache and not response.cached:
                self._cache.set(query, response)

            return ToolResult.success(
                response.to_formatted_text(),
                result_count=len(response.results),
                cached=response.cached,
                search_time_ms=response.search_time_ms,
            )

        except Exception as e:
            logger.exception(f"Search error for query: {query}")
            return ToolResult.error(f"Search error: {e}")

    async def _execute_search(self, query: str, count: int) -> SearchResponse:
        """Execute search in sandbox and parse response.

        Args:
            query: Search query.
            count: Number of results.

        Returns:
            Parsed SearchResponse.

        Raises:
            Exception: On search failure.
        """
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
            raise TimeoutError("Search request timed out")

        # Parse JSON output
        output = result.stdout.strip() if result.stdout else ""
        if not output:
            raise ValueError("Empty response from search")

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        # Check for error response
        if "error" in data:
            error_code = data.get("code", 0)
            error_msg = data["error"]
            raise Exception(f"{error_msg} (code: {error_code})")

        # Parse into SearchResponse
        return SearchResponse.from_json(output)

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._executor:
            await self._executor.cleanup()
