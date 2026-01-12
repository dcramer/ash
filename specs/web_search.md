# Web Search

> Search the web via Brave Search API with caching, retry, and structured output

Files: src/ash/tools/builtin/web_search.py, src/ash/tools/builtin/search_types.py,
       src/ash/tools/builtin/search_cache.py, src/ash/tools/retry.py

## Requirements

### MUST

- Execute search requests inside Docker sandbox
- Require network_mode: bridge (error if none)
- Pass API key via environment variable (not command line)
- URL-encode query parameters properly
- Return structured SearchResponse with citation metadata
- Cache search results (15 min TTL, 100 max entries)
- Retry on transient errors (429, 5xx) with exponential backoff
- NOT retry on auth errors (401) or bad requests (400)
- Accurately count results regardless of result number (1-100+)
- Truncate descriptions at word boundaries, not mid-word
- Handle HTTP errors gracefully
- Handle timeout (30s default)
- Respect sandbox proxy settings when configured

### SHOULD

- Limit results count (default 5, max 10)
- Include site_name extracted from URL domain
- Include published_date when available from API
- Log retry attempts with delay information
- Normalize cache keys (lowercase, strip whitespace)
- Include search metadata in response

### MAY

- Support output_format parameter (json, text)
- Include additional Brave API fields (favicon, thumbnail)
- Provide cache statistics via metadata
- Support additional search providers

## Interface

```python
@dataclass
class SearchResult:
    title: str
    url: str
    description: str
    site_name: str | None = None
    published_date: str | None = None

    def to_citation(self, index: int) -> str: ...

@dataclass
class SearchResponse:
    query: str
    results: list[SearchResult]
    total_results: int
    search_time_ms: int
    cached: bool = False

    def to_json(self) -> str: ...
    def to_formatted_text(self) -> str: ...

class SearchCache:
    def __init__(self, maxsize: int = 100, ttl: int = 900): ...
    def get(self, key: str) -> SearchResponse | None: ...
    def set(self, key: str, value: SearchResponse) -> None: ...
    def invalidate(self, key: str | None = None) -> None: ...

class WebSearchTool(Tool):
    name = "web_search"

    def __init__(
        self,
        api_key: str,
        sandbox_config: SandboxConfig,
        workspace_path: Path | None = None,
        cache: SearchCache | None = None,
        retry_config: RetryConfig | None = None,
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
| `{"query": "python async"}` | SearchResponse JSON | Structured results |
| `{"query": "test", "count": 3}` | 3 results | Limited count |
| Repeat query within 15 min | Cached response | `cached: true` in metadata |
| Empty query | Error: "Query required" | Validation |
| Network timeout | Retry up to 3 times | Exponential backoff |
| HTTP 429 rate limit | Retry with backoff | Up to 3 attempts |
| HTTP 401 auth error | Immediate error | No retry |

## Errors

| Condition | Response |
|-----------|----------|
| network_mode: none | ToolResult.error("Web search requires network_mode: bridge") |
| Missing API key | ToolResult.error("Brave Search API key not configured") |
| HTTP 401 | ToolResult.error("Invalid API key") |
| HTTP 429 after retries | ToolResult.error("Rate limit exceeded after 3 attempts") |
| Timeout after retries | ToolResult.error("Search request timed out after 3 attempts") |
| No results | ToolResult.success with result_count: 0 |

## Verification

```bash
uv run pytest tests/test_tools.py -v -k web_search
uv run pytest tests/test_search_cache.py -v
```

- Search executes in sandbox container
- API key not visible in command line (check ps/logs)
- Proxy settings respected when configured
- Proper error on network_mode: none
- Results formatted correctly with citation support
- Cache hit on repeated queries
- Retry on transient errors
- No retry on auth errors
