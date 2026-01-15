"""Structured types for web search results."""

import json
from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass
class SearchResult:
    """Individual search result with citation metadata."""

    title: str
    url: str
    description: str
    site_name: str | None = None
    published_date: str | None = None

    @classmethod
    def from_dict(cls, data: dict) -> "SearchResult":
        """Create from dictionary (e.g., parsed JSON)."""
        return cls(
            title=data.get("title", ""),
            url=data.get("url", ""),
            description=data.get("description", ""),
            site_name=data.get("site_name"),
            published_date=data.get("published_date"),
        )

    def to_citation(self, index: int) -> str:
        """Format as citation: [1] Title - site.com."""
        site = self.site_name or urlparse(self.url).netloc
        return f"[{index}] {self.title} - {site}"


@dataclass
class SearchResponse:
    """Complete search response with metadata."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    total_results: int = 0
    search_time_ms: int = 0
    cached: bool = False
    search_type: str = "web"

    @classmethod
    def from_json(cls, json_str: str) -> "SearchResponse":
        """Parse from JSON string."""
        data = json.loads(json_str)
        results = [SearchResult.from_dict(r) for r in data.get("results", [])]
        return cls(
            query=data.get("query", ""),
            results=results,
            total_results=data.get("total_count", len(results)),
            search_time_ms=data.get("search_time_ms", 0),
            search_type=data.get("search_type", "web"),
        )

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(
            {
                "query": self.query,
                "results": [
                    {
                        "title": r.title,
                        "url": r.url,
                        "description": r.description,
                        "site_name": r.site_name,
                        "published_date": r.published_date,
                    }
                    for r in self.results
                ],
                "total_count": self.total_results,
                "search_time_ms": self.search_time_ms,
                "cached": self.cached,
                "search_type": self.search_type,
            },
            indent=2,
        )

    def to_formatted_text(self) -> str:
        """Format as human-readable text."""
        if not self.results:
            return f"No results found for: {self.query}"

        lines = []
        for i, result in enumerate(self.results, 1):
            lines.append(f"{i}. {result.title}")
            lines.append(f"   URL: {result.url}")
            if result.description:
                lines.append(f"   {result.description}")
            lines.append("")
        return "\n".join(lines).strip()

    def get_citations(self) -> list[str]:
        """Get formatted citation list."""
        return [r.to_citation(i) for i, r in enumerate(self.results, 1)]
