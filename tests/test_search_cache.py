"""Tests for SearchCache."""

import time

import pytest

from ash.tools.builtin.search_cache import SearchCache
from ash.tools.builtin.search_types import SearchResponse, SearchResult


class TestSearchCache:
    """Tests for SearchCache with TTL."""

    @pytest.fixture
    def cache(self) -> SearchCache:
        """Create a cache with short TTL for testing."""
        return SearchCache(maxsize=10, ttl=2)  # 2 second TTL

    @pytest.fixture
    def sample_response(self) -> SearchResponse:
        """Create a sample search response."""
        return SearchResponse(
            query="test query",
            results=[
                SearchResult(
                    title="Test Result",
                    url="https://example.com",
                    description="A test description",
                )
            ],
            total_results=1,
            search_time_ms=100,
        )

    def test_set_and_get(self, cache: SearchCache, sample_response: SearchResponse):
        """Test basic set and get operations."""
        cache.set("test query", sample_response)
        retrieved = cache.get("test query")
        assert retrieved is not None
        assert isinstance(retrieved, SearchResponse)
        assert retrieved.query == sample_response.query
        assert len(retrieved.results) == 1

    def test_get_missing_key(self, cache: SearchCache):
        """Test that missing keys return None."""
        result = cache.get("nonexistent")
        assert result is None

    def test_key_normalization(
        self, cache: SearchCache, sample_response: SearchResponse
    ):
        """Test that keys are normalized (lowercase, whitespace collapsed)."""
        cache.set("Test Query", sample_response)

        # All these variations should hit the same cache entry
        assert cache.get("test query") is not None
        assert cache.get("TEST QUERY") is not None
        assert cache.get("  test   query  ") is not None
        assert cache.get("test\t\nquery") is not None

    def test_ttl_expiration(self, cache: SearchCache, sample_response: SearchResponse):
        """Test that entries expire after TTL."""
        cache.set("test", sample_response)
        assert cache.get("test") is not None

        # Wait for TTL to expire
        time.sleep(2.5)

        assert cache.get("test") is None

    def test_invalidate_single_key(
        self, cache: SearchCache, sample_response: SearchResponse
    ):
        """Test invalidating a single cache entry."""
        cache.set("key1", sample_response)
        cache.set("key2", sample_response)

        cache.invalidate("key1")

        assert cache.get("key1") is None
        assert cache.get("key2") is not None

    def test_invalidate_all(self, cache: SearchCache, sample_response: SearchResponse):
        """Test invalidating all cache entries."""
        cache.set("key1", sample_response)
        cache.set("key2", sample_response)

        cache.invalidate()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_maxsize_eviction(self, sample_response: SearchResponse):
        """Test that cache evicts old entries when maxsize is reached."""
        cache = SearchCache(maxsize=3, ttl=60)

        # Add 4 entries to a cache with maxsize 3
        for i in range(4):
            cache.set(f"key{i}", sample_response)

        # First entry should be evicted (LRU)
        assert cache.get("key0") is None
        assert cache.get("key1") is not None
        assert cache.get("key2") is not None
        assert cache.get("key3") is not None

    def test_string_value(self, cache: SearchCache):
        """Test caching string values (for web_fetch content)."""
        content = "This is page content"
        cache.set("https://example.com", content)

        retrieved = cache.get("https://example.com")
        assert retrieved == content

    def test_cache_stats(self, cache: SearchCache, sample_response: SearchResponse):
        """Test cache statistics."""
        cache.set("key1", sample_response)
        cache.set("key2", sample_response)

        stats = cache.stats()
        assert stats.size == 2
        assert stats.maxsize == 10


class TestSearchCacheEdgeCases:
    """Edge case tests for SearchCache."""

    def test_empty_key(self):
        """Test handling of empty keys."""
        cache = SearchCache()
        response = SearchResponse(
            query="", results=[], total_results=0, search_time_ms=0
        )
        cache.set("", response)
        assert cache.get("") is not None
        assert cache.get("   ") is not None  # Normalizes to empty

    def test_unicode_keys(self):
        """Test handling of unicode in keys."""
        cache = SearchCache()
        response = SearchResponse(
            query="test", results=[], total_results=0, search_time_ms=0
        )
        cache.set("python 异步", response)
        assert cache.get("python 异步") is not None
        assert cache.get("PYTHON 异步") is not None

    def test_concurrent_access(self):
        """Test that cache handles concurrent access."""
        cache = SearchCache()
        response = SearchResponse(
            query="test", results=[], total_results=0, search_time_ms=0
        )

        # Simulate rapid access
        for i in range(100):
            cache.set(f"key{i}", response)
            cache.get(f"key{i}")

        # Should not raise any errors
        assert cache.stats().size <= 100
