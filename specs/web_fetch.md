# Web Fetch

> Fetch and extract content from URLs, executed in sandbox

Files: src/ash/tools/builtin/web_fetch.py

## Requirements

### MUST

- Execute HTTP requests inside Docker sandbox
- Require network_mode: bridge (error if none)
- Support HTTP and HTTPS URLs
- Extract readable text content from HTML pages
- Remove script, style, and other non-content elements
- Handle HTTP redirects (up to 5 hops)
- Report final URL after redirects
- Respect timeout (30s default)
- Truncate content at max_length parameter
- Return structured JSON response with metadata
- Cache fetched content (15 min TTL)
- Set appropriate User-Agent header

### SHOULD

- Convert HTML structure to markdown-like format
- Preserve links as markdown `[text](url)` format
- Preserve headings as markdown `#` format
- Preserve lists as markdown bullet format
- Include page title in response
- Handle common content types (HTML, JSON, plain text)
- Report content truncation in metadata

### MAY

- Extract meta description and author
- Handle non-UTF8 encodings gracefully
- Support custom timeout per request
- Respect robots.txt (configurable)

## Interface

```python
class WebFetchTool(Tool):
    name = "web_fetch"

    def __init__(
        self,
        sandbox_config: SandboxConfig,
        workspace_path: Path | None = None,
        cache: SearchCache | None = None,
        max_length: int = 50000,
        timeout: int = 30,
    ): ...

    async def execute(
        self,
        input_data: {
            "url": str,
            "extract_mode": "text" | "markdown" = "markdown",
            "max_length": int = 50000
        },
        context: ToolContext,
    ) -> ToolResult: ...
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `{"url": "https://example.com"}` | Page content in markdown | Default mode |
| `{"url": "...", "extract_mode": "text"}` | Plain text only | No formatting |
| URL with redirects | Content from final URL | `final_url` in metadata |
| Repeat URL within 15 min | Cached content | `cached: true` |
| Very long page | Truncated content | `truncated: true` |
| Invalid URL scheme | Error | Only http/https |
| Non-HTML content type | Raw text or JSON | Content-type detection |

## Errors

| Condition | Response |
|-----------|----------|
| network_mode: none | ToolResult.error("Web fetch requires network_mode: bridge") |
| Invalid URL | ToolResult.error("Invalid URL: must be http or https") |
| HTTP 404 | ToolResult.error("Page not found (404)") |
| HTTP 403 | ToolResult.error("Access forbidden (403)") |
| Timeout | ToolResult.error("Request timed out after 30s") |
| Too many redirects | ToolResult.error("Too many redirects (max 5)") |
| Connection error | ToolResult.error("Failed to connect: {reason}") |

## Verification

```bash
uv run pytest tests/test_web_fetch.py -v
```

- Fetch executes in sandbox container
- HTML converted to readable markdown
- Links and headings preserved
- Content truncated at max_length
- Cache hit on repeated URLs
- Proper error on network_mode: none
- Redirect chain followed correctly
