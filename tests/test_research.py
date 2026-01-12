"""Tests for research skill."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.skills.research import (
    DEPTH_CONFIGS,
    ResearchConfig,
    ResearchSource,
    build_query_generation_prompt,
    build_synthesis_prompt,
    calculate_relevance_score,
    dedupe_and_rank_sources,
    parse_queries_response,
    parse_search_results,
)


class TestResearchSource:
    """Tests for ResearchSource dataclass."""

    def test_domain_extraction(self):
        """Test that domain is extracted from URL."""
        source = ResearchSource(
            url="https://www.example.com/page",
            title="Test",
            snippet="A test snippet",
        )
        assert source.domain == "example.com"

    def test_domain_without_www(self):
        """Test domain extraction without www prefix."""
        source = ResearchSource(
            url="https://docs.python.org/3/library/",
            title="Python Docs",
            snippet="Documentation",
        )
        assert source.domain == "docs.python.org"

    def test_domain_with_explicit_value(self):
        """Test that explicit domain is not overwritten."""
        source = ResearchSource(
            url="https://example.com",
            title="Test",
            snippet="Snippet",
            domain="custom.domain",
        )
        assert source.domain == "custom.domain"

    def test_invalid_url_domain(self):
        """Test handling of invalid URL for domain extraction."""
        source = ResearchSource(
            url="not-a-valid-url",
            title="Test",
            snippet="Snippet",
        )
        # Should have empty domain or the invalid URL
        assert source.domain == ""


class TestDepthConfigs:
    """Tests for depth level configurations."""

    def test_quick_config(self):
        """Test quick depth configuration."""
        config = DEPTH_CONFIGS["quick"]
        assert config.queries == 2
        assert config.sources_to_fetch == 3
        assert config.max_per_domain == 3

    def test_standard_config(self):
        """Test standard depth configuration."""
        config = DEPTH_CONFIGS["standard"]
        assert config.queries == 5
        assert config.sources_to_fetch == 10

    def test_deep_config(self):
        """Test deep depth configuration."""
        config = DEPTH_CONFIGS["deep"]
        assert config.queries == 10
        assert config.sources_to_fetch == 20


class TestQueryGeneration:
    """Tests for query generation utilities."""

    def test_build_query_generation_prompt_basic(self):
        """Test basic query generation prompt."""
        prompt = build_query_generation_prompt("machine learning", 5, None)
        assert "machine learning" in prompt
        assert "5" in prompt
        assert "JSON array" in prompt

    def test_build_query_generation_prompt_with_focus(self):
        """Test query generation prompt with focus area."""
        prompt = build_query_generation_prompt("machine learning", 5, "neural networks")
        assert "neural networks" in prompt
        assert "machine learning" in prompt

    def test_parse_queries_response_json_array(self):
        """Test parsing JSON array response."""
        response = '["query 1", "query 2", "query 3"]'
        queries = parse_queries_response(response, 5)
        assert len(queries) == 3
        assert queries[0] == "query 1"

    def test_parse_queries_response_with_extra_text(self):
        """Test parsing response with extra text around JSON."""
        response = 'Here are the queries:\n["query 1", "query 2"]\nThese should work.'
        queries = parse_queries_response(response, 5)
        assert len(queries) == 2

    def test_parse_queries_response_fallback(self):
        """Test fallback parsing for non-JSON response."""
        response = """1. first query here
2. second query here
3. third query"""
        queries = parse_queries_response(response, 5)
        assert len(queries) == 3
        assert "first query" in queries[0]

    def test_parse_queries_response_limit(self):
        """Test that query count is limited."""
        response = '["q1", "q2", "q3", "q4", "q5", "q6"]'
        queries = parse_queries_response(response, 3)
        assert len(queries) == 3


class TestRelevanceScoring:
    """Tests for relevance score calculation."""

    def test_base_score(self):
        """Test base relevance score."""
        source = ResearchSource(
            url="https://random-site.com",
            title="",
            snippet="",
        )
        score = calculate_relevance_score(source)
        assert score == 0.5  # Base score only

    def test_edu_domain_bonus(self):
        """Test .edu domain gets bonus."""
        source = ResearchSource(
            url="https://mit.edu/article",
            title="MIT Article",
            snippet="Educational content",
        )
        score = calculate_relevance_score(source)
        assert score > 0.5  # Should have authority bonus

    def test_gov_domain_bonus(self):
        """Test .gov domain gets highest bonus."""
        source = ResearchSource(
            url="https://cdc.gov/health",
            title="CDC Health",
            snippet="Government health info",
        )
        score = calculate_relevance_score(source)
        assert score > 0.5

    def test_long_snippet_bonus(self):
        """Test long snippet gets bonus."""
        source = ResearchSource(
            url="https://example.com",
            title="Test",
            snippet="A" * 250,  # Long snippet
        )
        score = calculate_relevance_score(source)
        assert score > 0.5  # Has snippet length bonus

    def test_title_bonus(self):
        """Test having a title gets bonus."""
        source = ResearchSource(
            url="https://example.com",
            title="Real Title Here",
            snippet="Some content",
        )
        score = calculate_relevance_score(source)
        assert score > 0.5  # Has title bonus

    def test_score_capped_at_1(self):
        """Test that score doesn't exceed 1.0."""
        source = ResearchSource(
            url="https://docs.python.org",
            title="Python Documentation",
            snippet="A" * 300,  # Long snippet
        )
        score = calculate_relevance_score(source)
        assert score <= 1.0


class TestSourceDeduplication:
    """Tests for source deduplication and ranking."""

    @pytest.fixture
    def config(self) -> ResearchConfig:
        """Create a test config."""
        return ResearchConfig(queries=5, sources_to_fetch=10, max_per_domain=2)

    def test_url_deduplication(self, config):
        """Test that duplicate URLs are removed."""
        sources = [
            ResearchSource(
                url="https://example.com/page",
                title="Page 1",
                snippet="First",
            ),
            ResearchSource(
                url="https://example.com/page",
                title="Page 1 Duplicate",
                snippet="Second",
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        assert len(result) == 1

    def test_url_deduplication_case_insensitive(self, config):
        """Test URL dedup is case-insensitive."""
        sources = [
            ResearchSource(
                url="https://EXAMPLE.COM/Page",
                title="Page 1",
                snippet="First",
            ),
            ResearchSource(
                url="https://example.com/page",
                title="Page 2",
                snippet="Second",
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        assert len(result) == 1

    def test_title_similarity_deduplication(self, config):
        """Test that similar titles are deduplicated."""
        sources = [
            ResearchSource(
                url="https://site1.com",
                title="Introduction to Python Programming",
                snippet="First",
            ),
            ResearchSource(
                url="https://site2.com",
                title="Introduction to Python Programming 2024",
                snippet="Second",
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        # Titles are >85% similar, should be deduped
        assert len(result) == 1

    def test_different_titles_not_deduped(self, config):
        """Test that different titles are kept."""
        sources = [
            ResearchSource(
                url="https://site1.com",
                title="Python Basics",
                snippet="First",
            ),
            ResearchSource(
                url="https://site2.com",
                title="JavaScript Fundamentals",
                snippet="Second",
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        assert len(result) == 2

    def test_domain_limit(self, config):
        """Test that sources per domain are limited."""
        sources = [
            ResearchSource(
                url="https://example.com/page1",
                title="Page 1",
                snippet="First",
            ),
            ResearchSource(
                url="https://example.com/page2",
                title="Page 2",
                snippet="Second",
            ),
            ResearchSource(
                url="https://example.com/page3",
                title="Page 3",
                snippet="Third",
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        # max_per_domain is 2
        assert len(result) == 2

    def test_sorted_by_relevance(self, config):
        """Test that results are sorted by relevance score."""
        sources = [
            ResearchSource(
                url="https://random.com",
                title="Low relevance",
                snippet="Short",
            ),
            ResearchSource(
                url="https://docs.python.org",
                title="Python Documentation",
                snippet="A" * 250,
            ),
        ]
        result = dedupe_and_rank_sources(sources, config)
        # Higher relevance (docs. domain + long snippet) should be first
        assert "python" in result[0].url.lower()


class TestSearchResultParsing:
    """Tests for parsing search results."""

    def test_parse_numbered_format(self):
        """Test parsing numbered result format."""
        content = """1. Python Documentation
   URL: https://docs.python.org
   Official Python documentation

2. Real Python
   URL: https://realpython.com
   Python tutorials and guides"""

        sources = parse_search_results(content)
        assert len(sources) == 2
        assert sources[0].title == "Python Documentation"
        assert sources[0].url == "https://docs.python.org"
        assert "Official Python" in sources[0].snippet

    def test_parse_multiline_description(self):
        """Test parsing multi-line descriptions."""
        content = """1. Test Page
   URL: https://example.com
   This is a long description
   that spans multiple lines"""

        sources = parse_search_results(content)
        assert len(sources) == 1
        assert "spans multiple lines" in sources[0].snippet

    def test_parse_empty_content(self):
        """Test parsing empty content."""
        sources = parse_search_results("")
        assert len(sources) == 0


class TestSynthesisPrompt:
    """Tests for synthesis prompt generation."""

    def test_build_synthesis_prompt_basic(self):
        """Test basic synthesis prompt."""
        sources = [
            ResearchSource(
                url="https://example.com",
                title="Test Source",
                snippet="Test snippet",
                content="Full content here",
            )
        ]
        prompt = build_synthesis_prompt("test topic", sources, None)
        assert "test topic" in prompt
        assert "Test Source" in prompt
        assert "Full content here" in prompt

    def test_build_synthesis_prompt_with_focus(self):
        """Test synthesis prompt with focus."""
        sources = [
            ResearchSource(
                url="https://example.com",
                title="Source",
                snippet="Snippet",
            )
        ]
        prompt = build_synthesis_prompt("topic", sources, "specific aspect")
        assert "specific aspect" in prompt

    def test_build_synthesis_prompt_truncates_content(self):
        """Test that long content is truncated."""
        sources = [
            ResearchSource(
                url="https://example.com",
                title="Source",
                snippet="Snippet",
                content="A" * 5000,  # Very long content
            )
        ]
        prompt = build_synthesis_prompt("topic", sources, None)
        # Content should be truncated to ~3000 chars
        assert prompt.count("A") <= 3500  # Some margin for formatting

    def test_build_synthesis_prompt_uses_snippet_fallback(self):
        """Test that snippet is used when content is None."""
        sources = [
            ResearchSource(
                url="https://example.com",
                title="Source",
                snippet="The snippet content",
                content=None,
            )
        ]
        prompt = build_synthesis_prompt("topic", sources, None)
        assert "The snippet content" in prompt


class TestResearchIntegration:
    """Integration tests for research workflow (with mocks)."""

    @pytest.fixture
    def mock_tool_executor(self):
        """Create a mock tool executor."""
        executor = AsyncMock()

        # Mock web_search results
        async def mock_execute(tool_name, params, context):
            if tool_name == "web_search":
                return MagicMock(
                    is_error=False,
                    content="""1. Result One
   URL: https://example.com/1
   Description one

2. Result Two
   URL: https://other.com/2
   Description two""",
                )
            if tool_name == "web_fetch":
                return MagicMock(
                    is_error=False,
                    content="Fetched page content here",
                )
            return MagicMock(is_error=True, content="Unknown tool")

        executor.execute = mock_execute
        return executor

    @pytest.fixture
    def mock_llm_provider(self):
        """Create a mock LLM provider."""
        provider = AsyncMock()

        async def mock_complete(**kwargs):
            response = MagicMock()
            # Check if this is query generation or synthesis
            if "Generate" in kwargs.get("system", ""):
                response.message.get_text.return_value = '["query 1", "query 2"]'
            else:
                response.message.get_text.return_value = (
                    "# Research Report\n\nSynthesized findings [1][2]"
                )
            return response

        provider.complete = mock_complete
        return provider
