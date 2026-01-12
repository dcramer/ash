"""Web fetch tool for URL content extraction, executed in sandbox."""

import json
import logging
import shlex
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from ash.sandbox import SandboxExecutor
from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig
from ash.tools.base import Tool, ToolContext, ToolResult
from ash.tools.builtin.search_cache import SearchCache

if TYPE_CHECKING:
    from ash.config.models import SandboxConfig

logger = logging.getLogger(__name__)

# Python script to execute inside sandbox for fetching URLs
# Uses stdlib only - html.parser for HTML extraction
FETCH_SCRIPT = '''
import json, os, sys, urllib.request, urllib.error
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

url = sys.argv[1]
extract_mode = sys.argv[2] if len(sys.argv) > 2 else "markdown"
max_length = int(sys.argv[3]) if len(sys.argv) > 3 else 50000

# Validate URL
parsed = urlparse(url)
if parsed.scheme not in ("http", "https"):
    print(json.dumps({"error": "Invalid URL: must be http or https", "code": 400}))
    sys.exit(1)

class ContentExtractor(HTMLParser):
    """Extract readable content from HTML."""

    def __init__(self, base_url, mode="markdown"):
        super().__init__()
        self.base_url = base_url
        self.mode = mode
        self.content = []
        self.in_skip = 0  # Counter for nested skip elements
        self.in_title = False
        self.title = ""
        self.skip_tags = {"script", "style", "noscript", "nav", "footer", "header", "aside"}
        self.heading_tags = {"h1", "h2", "h3", "h4", "h5", "h6"}
        self.current_tag = None
        self.current_link_href = None
        self.current_link_text = []

    def handle_starttag(self, tag, attrs):
        self.current_tag = tag
        attrs_dict = dict(attrs)

        if tag in self.skip_tags:
            self.in_skip += 1
            return

        if tag == "title":
            self.in_title = True
            return

        if self.in_skip:
            return

        if self.mode == "markdown":
            if tag in self.heading_tags:
                level = int(tag[1])
                self.content.append("\\n" + "#" * level + " ")
            elif tag == "p":
                self.content.append("\\n\\n")
            elif tag == "br":
                self.content.append("\\n")
            elif tag == "li":
                self.content.append("\\n- ")
            elif tag == "a":
                href = attrs_dict.get("href", "")
                if href and not href.startswith("#"):
                    self.current_link_href = urljoin(self.base_url, href)
                    self.current_link_text = []
            elif tag in ("ul", "ol"):
                self.content.append("\\n")
            elif tag == "blockquote":
                self.content.append("\\n> ")
            elif tag == "code":
                self.content.append("`")
            elif tag == "pre":
                self.content.append("\\n```\\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self.in_skip = max(0, self.in_skip - 1)
            return

        if tag == "title":
            self.in_title = False
            return

        if self.in_skip:
            return

        if self.mode == "markdown":
            if tag == "a" and self.current_link_href:
                link_text = "".join(self.current_link_text).strip()
                if link_text:
                    self.content.append(f"[{link_text}]({self.current_link_href})")
                self.current_link_href = None
                self.current_link_text = []
            elif tag == "code":
                self.content.append("`")
            elif tag == "pre":
                self.content.append("\\n```\\n")
            elif tag in self.heading_tags:
                self.content.append("\\n")

        self.current_tag = None

    def handle_data(self, data):
        if self.in_title:
            self.title += data
            return

        if self.in_skip:
            return

        text = data.strip()
        if not text:
            return

        if self.current_link_href:
            self.current_link_text.append(data)
        else:
            # Normalize whitespace
            text = " ".join(data.split())
            self.content.append(text)

    def get_content(self):
        content = "".join(self.content)
        # Clean up excessive newlines
        while "\\n\\n\\n" in content:
            content = content.replace("\\n\\n\\n", "\\n\\n")
        return content.strip()

# Fetch the URL
headers = {
    "User-Agent": "Mozilla/5.0 (compatible; AshBot/1.0; +https://github.com/dcramer/ash)",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

final_url = url
redirect_count = 0
max_redirects = 5

try:
    while redirect_count < max_redirects:
        req = urllib.request.Request(final_url, headers=headers)
        opener = urllib.request.build_opener(urllib.request.HTTPRedirectHandler())

        with urllib.request.urlopen(req, timeout=30) as resp:
            if resp.status in (301, 302, 303, 307, 308):
                new_url = resp.headers.get("Location")
                if new_url:
                    final_url = urljoin(final_url, new_url)
                    redirect_count += 1
                    continue

            content_type = resp.headers.get("Content-Type", "")

            # Handle non-HTML content
            if "application/json" in content_type:
                raw_content = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed_json = json.loads(raw_content)
                    content = json.dumps(parsed_json, indent=2)
                except json.JSONDecodeError:
                    content = raw_content
                output = {
                    "url": url,
                    "final_url": final_url,
                    "title": None,
                    "content": content[:max_length],
                    "content_type": content_type,
                    "status_code": resp.status,
                    "truncated": len(content) > max_length,
                }
                print(json.dumps(output))
                sys.exit(0)

            if "text/plain" in content_type:
                content = resp.read().decode("utf-8", errors="replace")
                output = {
                    "url": url,
                    "final_url": final_url,
                    "title": None,
                    "content": content[:max_length],
                    "content_type": content_type,
                    "status_code": resp.status,
                    "truncated": len(content) > max_length,
                }
                print(json.dumps(output))
                sys.exit(0)

            # HTML content
            raw_html = resp.read().decode("utf-8", errors="replace")
            break
    else:
        print(json.dumps({"error": "Too many redirects (max 5)", "code": 310}))
        sys.exit(1)

except urllib.error.HTTPError as e:
    error_msgs = {
        403: "Access forbidden (403)",
        404: "Page not found (404)",
        500: "Server error (500)",
    }
    print(json.dumps({
        "error": error_msgs.get(e.code, f"HTTP {e.code}"),
        "code": e.code
    }))
    sys.exit(1)
except urllib.error.URLError as e:
    print(json.dumps({"error": f"Failed to connect: {e.reason}", "code": 0}))
    sys.exit(1)
except TimeoutError:
    print(json.dumps({"error": "Request timed out after 30s", "code": 408}))
    sys.exit(1)
except Exception as e:
    print(json.dumps({"error": str(e), "code": 0}))
    sys.exit(1)

# Extract content
extractor = ContentExtractor(final_url, mode=extract_mode)
try:
    extractor.feed(raw_html)
except Exception as e:
    print(json.dumps({"error": f"Failed to parse HTML: {e}", "code": 0}))
    sys.exit(1)

content = extractor.get_content()
title = extractor.title.strip() or None

# Truncate if needed
truncated = len(content) > max_length
if truncated:
    content = content[:max_length]
    # Try to truncate at word boundary
    last_space = content.rfind(" ")
    if last_space > max_length * 0.9:
        content = content[:last_space]
    content = content.rstrip() + "..."

output = {
    "url": url,
    "final_url": final_url,
    "title": title,
    "content": content,
    "content_type": content_type,
    "status_code": resp.status,
    "truncated": truncated,
}

print(json.dumps(output))
'''


class WebFetchTool(Tool):
    """Fetch and extract content from URLs.

    All requests execute inside the Docker sandbox for network control.
    Requires network_mode: bridge in sandbox configuration.

    Features:
    - Extracts readable content from HTML pages
    - Converts to markdown-like format (links, headings, lists)
    - Handles redirects (up to 5 hops)
    - Caches fetched content (30 min TTL by default)
    - Supports text/HTML/JSON content types
    """

    def __init__(
        self,
        sandbox_config: "SandboxConfig | None" = None,
        workspace_path: Path | None = None,
        cache: SearchCache | None = None,
        max_length: int = 50000,
        timeout: int = 30,
    ):
        """Initialize web fetch tool.

        Args:
            sandbox_config: Sandbox configuration (pydantic model from config).
            workspace_path: Path to workspace (for sandbox config).
            cache: Optional cache for fetched content.
            max_length: Maximum content length to return.
            timeout: Request timeout in seconds.
        """
        self._sandbox_config = sandbox_config
        self._cache = cache
        self._max_length = max_length
        self._timeout = timeout

        # Check network mode
        network_mode = sandbox_config.network_mode if sandbox_config else "bridge"
        if network_mode == "none":
            raise ValueError(
                "Web fetch requires network_mode: bridge in sandbox configuration"
            )

        # Build sandbox config
        manager_config = self._build_manager_config(sandbox_config, workspace_path)
        self._executor = SandboxExecutor(config=manager_config)

    def _build_manager_config(
        self,
        config: "SandboxConfig | None",
        workspace_path: Path | None,
    ) -> SandboxManagerConfig:
        """Convert pydantic SandboxConfig to manager's dataclass config."""
        if config is None:
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
        return "web_fetch"

    @property
    def description(self) -> str:
        return (
            "Fetch and read the content of a web page. "
            "Extracts readable text from HTML pages, converting to markdown format. "
            "Use this to read full articles, documentation, or other web content."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (http or https).",
                },
                "extract_mode": {
                    "type": "string",
                    "enum": ["text", "markdown"],
                    "description": "Content format: 'markdown' preserves structure, 'text' is plain.",
                    "default": "markdown",
                },
                "max_length": {
                    "type": "integer",
                    "description": f"Maximum content length (default {self._max_length}).",
                    "default": self._max_length,
                },
            },
            "required": ["url"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Fetch and extract content from URL.

        Args:
            input_data: Must contain 'url' key.
            context: Execution context.

        Returns:
            Tool result with extracted content.
        """
        url = input_data.get("url", "").strip()
        if not url:
            return ToolResult.error("Missing required parameter: url")

        # Validate URL scheme
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https"):
            return ToolResult.error("Invalid URL: must be http or https")

        extract_mode = input_data.get("extract_mode", "markdown")
        max_length = input_data.get("max_length", self._max_length)

        # Check cache first
        if self._cache:
            cached = self._cache.get(url)
            if cached is not None and isinstance(cached, str):
                logger.debug(f"Cache hit for URL: {url}")
                return ToolResult.success(
                    cached,
                    cached=True,
                    url=url,
                )

        try:
            result = await self._fetch_url(url, extract_mode, max_length)

            # Cache the content
            if self._cache and "content" in result:
                self._cache.set(url, result["content"])

            return ToolResult.success(
                result.get("content", ""),
                cached=False,
                url=result.get("url", url),
                final_url=result.get("final_url", url),
                title=result.get("title"),
                content_type=result.get("content_type"),
                truncated=result.get("truncated", False),
            )

        except Exception as e:
            logger.exception(f"Fetch error for URL: {url}")
            return ToolResult.error(f"Fetch error: {e}")

    async def _fetch_url(
        self, url: str, extract_mode: str, max_length: int
    ) -> dict[str, Any]:
        """Fetch URL in sandbox and parse response.

        Args:
            url: URL to fetch.
            extract_mode: Content extraction mode.
            max_length: Maximum content length.

        Returns:
            Parsed response dict.

        Raises:
            Exception: On fetch failure.
        """
        escaped_url = shlex.quote(url)
        escaped_mode = shlex.quote(extract_mode)
        command = (
            f"python3 -c {shlex.quote(FETCH_SCRIPT)} "
            f"{escaped_url} {escaped_mode} {max_length}"
        )

        result = await self._executor.execute(
            command,
            timeout=self._timeout,
            reuse_container=True,
        )

        if result.timed_out:
            raise TimeoutError(f"Request timed out after {self._timeout}s")

        output = result.stdout.strip() if result.stdout else ""
        if not output:
            raise ValueError("Empty response from fetch")

        try:
            data = json.loads(output)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}") from e

        if "error" in data:
            raise Exception(data["error"])

        return data

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._executor:
            await self._executor.cleanup()
