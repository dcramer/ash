"""In-memory LRU cache with TTL for search results."""

import re
from dataclasses import dataclass

from cachetools import TTLCache

from ash.tools.builtin.search_types import SearchResponse


@dataclass
class CacheStats:
    """Cache statistics."""

    hits: int
    misses: int
    size: int
    maxsize: int


class SearchCache:
    """Thread-safe LRU cache with TTL for search results.

    Keys are normalized to ensure consistent cache hits:
    - Lowercased
    - Whitespace stripped and collapsed
    - Query type independent (can cache both search and fetch results)
    """

    def __init__(self, maxsize: int = 100, ttl: int = 900) -> None:
        """Initialize cache.

        Args:
            maxsize: Maximum number of entries (default 100).
            ttl: Time-to-live in seconds (default 900 = 15 min).
        """
        self._cache: TTLCache = TTLCache(maxsize=maxsize, ttl=ttl)
        self._hits = 0
        self._misses = 0

    @staticmethod
    def _normalize_key(key: str) -> str:
        """Normalize cache key for consistent matching.

        - Lowercase
        - Strip leading/trailing whitespace
        - Collapse multiple spaces to single space
        """
        return re.sub(r"\s+", " ", key.strip().lower())

    def get(self, key: str) -> SearchResponse | str | None:
        """Get cached response.

        Args:
            key: Cache key (will be normalized).

        Returns:
            Cached SearchResponse or string content, or None if not found.
        """
        normalized = self._normalize_key(key)
        result = self._cache.get(normalized)
        if result is not None:
            self._hits += 1
            # Mark as cached if it's a SearchResponse
            if isinstance(result, SearchResponse):
                result.cached = True
        else:
            self._misses += 1
        return result

    def set(self, key: str, value: SearchResponse | str) -> None:
        """Cache a response.

        Args:
            key: Cache key (will be normalized).
            value: SearchResponse or string content to cache.
        """
        normalized = self._normalize_key(key)
        self._cache[normalized] = value

    def invalidate(self, key: str | None = None) -> None:
        """Invalidate cache entries.

        Args:
            key: Specific key to invalidate, or None to clear all.
        """
        if key is None:
            self._cache.clear()
        else:
            normalized = self._normalize_key(key)
            self._cache.pop(normalized, None)

    def stats(self) -> CacheStats:
        """Get cache statistics."""
        return CacheStats(
            hits=self._hits,
            misses=self._misses,
            size=len(self._cache),
            maxsize=self._cache.maxsize,
        )
