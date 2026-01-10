# Web Search

> Search the web via Brave Search API, executed in sandbox

Files: src/ash/tools/builtin/web_search.py

## Requirements

### MUST

- Execute search requests inside Docker sandbox
- Require network_mode: bridge (error if none)
- Pass API key via environment variable (not command line)
- URL-encode query parameters properly
- Return formatted results with title, URL, description
- Handle HTTP errors gracefully
- Handle timeout (30s default)
- Respect sandbox proxy settings when configured

### SHOULD

- Limit results count (default 5, max 10)
- Truncate long descriptions
- Include search metadata in response

### MAY

- Cache recent results
- Support additional search providers

## Interface

```python
class WebSearchTool(Tool):
    name = "web_search"

    def __init__(
        self,
        api_key: str,
        sandbox_config: SandboxConfig,
        max_results: int = 10,
    ): ...

    async def execute(
        self,
        input_data: {"query": str, "count": int = 5},
        context: ToolContext,
    ) -> ToolResult: ...
```

## Configuration

```toml
[brave_search]
api_key = "..."  # or BRAVE_API_KEY env var

[sandbox]
network_mode = "bridge"  # Required for web_search
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| `{"query": "python async"}` | Formatted results | Success |
| `{"query": "test", "count": 3}` | 3 results | Limited |
| Empty query | Error: "Query required" | Validation |
| Network disabled | Error: "Network required" | Config check |
| API timeout | Error: "Search timed out" | 30s limit |
| Invalid API key | Error: "Authentication failed" | HTTP 401 |

## Errors

| Condition | Response |
|-----------|----------|
| network_mode: none | ToolResult.error("Web search requires network_mode: bridge") |
| Missing API key | ToolResult.error("Brave Search API key not configured") |
| HTTP 401 | ToolResult.error("Invalid API key") |
| HTTP 429 | ToolResult.error("Rate limit exceeded") |
| Timeout | ToolResult.error("Search request timed out") |
| No results | Empty result (not error) |

## Verification

```bash
uv run pytest tests/test_tools.py -v -k web_search
```

- Search executes in sandbox container
- API key not visible in command line (check ps/logs)
- Proxy settings respected when configured
- Proper error on network_mode: none
- Results formatted correctly
