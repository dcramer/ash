"""Web search tool using Brave Search API."""

from typing import Any

import httpx

from ash.tools.base import Tool, ToolContext, ToolResult

BRAVE_SEARCH_URL = "https://api.search.brave.com/res/v1/web/search"


class WebSearchTool(Tool):
    """Search the web using Brave Search API.

    Provides web search capabilities with snippets and URLs.
    """

    def __init__(
        self,
        api_key: str,
        max_results: int = 5,
    ):
        """Initialize web search tool.

        Args:
            api_key: Brave Search API key.
            max_results: Maximum results to return per search.
        """
        self._api_key = api_key
        self._max_results = max_results
        self._client = httpx.AsyncClient(timeout=30.0)

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
        """Execute web search.

        Args:
            input_data: Must contain 'query' key.
            context: Execution context.

        Returns:
            Tool result with search results.
        """
        query = input_data.get("query")
        if not query:
            return ToolResult.error("Missing required parameter: query")

        count = min(input_data.get("count", 5), self._max_results)

        try:
            response = await self._client.get(
                BRAVE_SEARCH_URL,
                params={
                    "q": query,
                    "count": count,
                },
                headers={
                    "Accept": "application/json",
                    "X-Subscription-Token": self._api_key,
                },
            )

            if response.status_code != 200:
                return ToolResult.error(
                    f"Search API error: {response.status_code} - {response.text}"
                )

            data = response.json()
            results = self._format_results(data)

            if not results:
                return ToolResult.success(
                    f"No results found for: {query}",
                    result_count=0,
                )

            return ToolResult.success(
                results,
                result_count=len(data.get("web", {}).get("results", [])),
            )

        except httpx.TimeoutException:
            return ToolResult.error("Search request timed out")
        except Exception as e:
            return ToolResult.error(f"Search error: {e}")

    def _format_results(self, data: dict[str, Any]) -> str:
        """Format search results as readable text.

        Args:
            data: Raw API response.

        Returns:
            Formatted search results.
        """
        web_results = data.get("web", {}).get("results", [])
        if not web_results:
            return ""

        lines = []
        for i, result in enumerate(web_results, 1):
            title = result.get("title", "No title")
            url = result.get("url", "")
            description = result.get("description", "No description")

            lines.append(f"{i}. {title}")
            lines.append(f"   URL: {url}")
            lines.append(f"   {description}")
            lines.append("")

        return "\n".join(lines).strip()

    async def close(self) -> None:
        """Close the HTTP client."""
        await self._client.aclose()
