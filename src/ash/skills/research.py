"""Research subagent - performs web research on topics.

This module provides both:
1. A simple subagent approach (build_subagent_config) for consistent execution
2. A programmatic approach (execute_research) for advanced multi-query orchestration

The subagent approach is preferred for consistency with other dynamic skills.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from ash.skills.base import SubagentConfig
    from ash.skills.registry import SkillRegistry

logger = logging.getLogger(__name__)

# Subagent configuration
ALLOWED_TOOLS = ["web_search", "web_fetch"]
MAX_ITERATIONS = 20

# Depth guidance for subagent prompt
DEPTH_GUIDANCE = {
    "quick": "Do 2-3 searches, read 2-3 sources. Be brief.",
    "standard": "Do 4-6 searches from different angles, read 5-8 sources.",
    "deep": "Do 8-10 searches covering all aspects, read 10-15 sources.",
}

# Input schema for the research skill
RESEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "topic": {
            "type": "string",
            "description": "The topic to research",
        },
        "depth": {
            "type": "string",
            "enum": ["quick", "standard", "deep"],
            "description": "Research depth (default: standard)",
        },
        "focus": {
            "type": "string",
            "description": "Optional specific aspect to focus on",
        },
    },
    "required": ["topic"],
}


def build_subagent_prompt(topic: str, depth: str, focus: str | None) -> str:
    """Build system prompt for research subagent.

    Args:
        topic: Research topic.
        depth: Research depth (quick, standard, deep).
        focus: Optional focus area.

    Returns:
        System prompt.
    """
    focus_text = f"\n\nFocus especially on: {focus}" if focus else ""

    return f"""# Research Assistant

Research the following topic thoroughly and produce a comprehensive report.

## Topic
{topic}{focus_text}

## Depth: {depth}
{DEPTH_GUIDANCE.get(depth, DEPTH_GUIDANCE["standard"])}

## Process

1. **Search**: Use web_search with varied queries to find diverse sources
2. **Read**: Use web_fetch to read the most relevant pages
3. **Synthesize**: Combine findings into a coherent report

## Output Format

Produce a research report with:
- **Summary**: 2-3 sentence executive summary
- **Findings**: Detailed analysis with inline citations [1], [2]
- **Sources**: Numbered list of sources used

## Guidelines

- Use multiple search queries to cover different aspects
- Prefer authoritative sources (.gov, .edu, official docs)
- Cite sources for all claims using [1], [2] notation
- Note any conflicting information between sources
- Be factual and objective"""


def build_subagent_config(
    input_data: dict[str, Any],
    **kwargs: Any,  # Accepts extra context (tool_definitions, workspace_path, etc.)
) -> "SubagentConfig":
    """Build SubagentConfig for research execution.

    Args:
        input_data: Input containing 'topic', optional 'depth' and 'focus'.
        **kwargs: Extra context from executor (ignored by research).

    Returns:
        SubagentConfig ready for execution.

    Raises:
        ValueError: If required input is missing.
    """
    from ash.skills.base import SubagentConfig

    topic = input_data.get("topic")
    if not topic:
        raise ValueError(
            "Missing required input: topic. Please specify what to research."
        )

    depth = input_data.get("depth", "standard")
    focus = input_data.get("focus")

    # Build system prompt
    system_prompt = build_subagent_prompt(topic, depth, focus)

    return SubagentConfig(
        system_prompt=system_prompt,
        allowed_tools=ALLOWED_TOOLS,
        max_iterations=MAX_ITERATIONS,
        initial_message="Research the topic and produce a comprehensive report.",
    )


def register(registry: "SkillRegistry") -> None:
    """Register the research skill with the registry.

    Args:
        registry: Skill registry to register with.
    """
    registry.register_dynamic(
        name="research",
        description="Research a topic using web search",
        build_config=build_subagent_config,
        required_tools=ALLOWED_TOOLS,
        input_schema=RESEARCH_INPUT_SCHEMA,
    )


# =============================================================================
# Programmatic research approach (alternative to subagent)
# =============================================================================


@dataclass
class ResearchSource:
    """A source found during research."""

    url: str
    title: str
    snippet: str
    content: str | None = None
    domain: str = ""
    relevance_score: float = 0.0

    def __post_init__(self) -> None:
        if not self.domain and self.url:
            try:
                parsed = urlparse(self.url)
                self.domain = parsed.netloc.lower()
                if self.domain.startswith("www."):
                    self.domain = self.domain[4:]
            except Exception:
                self.domain = ""  # Fallback to empty domain


@dataclass
class ResearchConfig:
    """Configuration for research depth levels."""

    queries: int
    sources_to_fetch: int
    max_per_domain: int = 3


# Depth level configurations
DEPTH_CONFIGS = {
    "quick": ResearchConfig(queries=2, sources_to_fetch=3),
    "standard": ResearchConfig(queries=5, sources_to_fetch=10),
    "deep": ResearchConfig(queries=10, sources_to_fetch=20),
}

# Authoritative domains get higher ranking
AUTHORITY_DOMAINS = {
    ".gov": 1.0,
    ".edu": 0.9,
    ".org": 0.7,
    "docs.": 0.8,
    "developer.": 0.8,
    "official": 0.6,
}


def build_query_generation_prompt(
    topic: str, num_queries: int, focus: str | None
) -> str:
    """Build prompt for LLM to generate search queries.

    Args:
        topic: The research topic.
        num_queries: Number of queries to generate.
        focus: Optional focus area.

    Returns:
        System prompt for query generation.
    """
    focus_instruction = ""
    if focus:
        focus_instruction = f"\nFocus especially on aspects related to: {focus}"

    return f"""You are a research query generator. Generate exactly {num_queries} diverse search queries to thoroughly research this topic:

Topic: {topic}{focus_instruction}

Generate queries that:
1. Cover different aspects of the topic
2. Use varied phrasing to find different sources
3. Include specific/technical queries and general ones
4. Target authoritative sources when relevant (e.g., "site:github.com", "official docs")

Output ONLY a JSON array of query strings, nothing else. Example:
["query 1", "query 2", "query 3"]"""


def build_synthesis_prompt(
    topic: str,
    sources: list[ResearchSource],
    focus: str | None,
) -> str:
    """Build prompt for LLM to synthesize research findings.

    Args:
        topic: The research topic.
        sources: Sources with content.
        focus: Optional focus area.

    Returns:
        System prompt for synthesis.
    """
    focus_instruction = ""
    if focus:
        focus_instruction = f"\nFocus especially on: {focus}"

    # Build source context
    source_blocks = []
    for i, source in enumerate(sources, 1):
        content = source.content or source.snippet
        # Limit content per source to avoid token overflow
        if len(content) > 3000:
            content = content[:3000] + "..."
        source_blocks.append(
            f"[Source {i}] {source.title}\nURL: {source.url}\nContent: {content}\n"
        )

    sources_text = "\n---\n".join(source_blocks)

    return f"""You are a research analyst. Synthesize the following sources into a comprehensive research report.

Topic: {topic}{focus_instruction}

SOURCES:
{sources_text}

Create a report with:
1. **Summary**: 2-3 sentence executive summary
2. **Findings**: Detailed analysis organized by subtopic, with inline citations [1], [2], etc.
3. **Methodology**: Note how many sources were analyzed
4. **Sources**: Numbered list with titles and URLs

Use inline citations [1], [2] throughout to attribute information to sources.
Note any conflicting information between sources.
Be factual and cite your sources for all claims."""


def parse_queries_response(response_text: str, expected_count: int) -> list[str]:
    """Parse LLM response to extract search queries.

    Args:
        response_text: LLM response text.
        expected_count: Expected number of queries.

    Returns:
        List of search queries.
    """
    # Try to find JSON array in response
    try:
        # Look for array pattern
        match = re.search(r"\[.*\]", response_text, re.DOTALL)
        if match:
            queries = json.loads(match.group())
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:expected_count]
    except (json.JSONDecodeError, TypeError):
        pass

    # Fallback: split by newlines and clean up
    lines = response_text.strip().split("\n")
    queries = []
    for line in lines:
        # Remove numbering, bullets, quotes
        cleaned = re.sub(r"^[\d\.\-\*\•]+\s*", "", line.strip())
        cleaned = cleaned.strip("\"'`")
        if cleaned and len(cleaned) > 5:
            queries.append(cleaned)

    return queries[:expected_count]


def calculate_relevance_score(source: ResearchSource) -> float:
    """Calculate relevance score for a source.

    Higher scores for:
    - Authoritative domains (.edu, .gov, docs.)
    - Longer, more detailed snippets
    - Titles that match search context

    Args:
        source: The source to score.

    Returns:
        Relevance score (0.0 to 1.0).
    """
    score = 0.5  # Base score

    # Authority bonus
    for pattern, bonus in AUTHORITY_DOMAINS.items():
        if pattern in source.domain.lower():
            score += bonus * 0.3
            break

    # Snippet length bonus (more content = likely more useful)
    if source.snippet:
        if len(source.snippet) > 200:
            score += 0.2
        elif len(source.snippet) > 100:
            score += 0.1

    # Title relevance (has content vs placeholder)
    if source.title and source.title.lower() not in ("untitled", "no title", ""):
        score += 0.1

    return min(1.0, score)


def dedupe_and_rank_sources(
    sources: list[ResearchSource],
    config: ResearchConfig,
) -> list[ResearchSource]:
    """Deduplicate and rank sources.

    - Remove exact URL duplicates
    - Remove near-duplicate titles (fuzzy match)
    - Limit sources per domain
    - Sort by relevance score

    Args:
        sources: Raw sources from search results.
        config: Research configuration.

    Returns:
        Deduplicated and ranked sources.
    """
    seen_urls: set[str] = set()
    seen_titles: list[str] = []
    domain_counts: dict[str, int] = {}
    unique_sources: list[ResearchSource] = []

    for source in sources:
        # Skip exact URL duplicates
        url_key = source.url.lower().rstrip("/")
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        # Skip near-duplicate titles (>85% similar)
        is_dupe_title = False
        for existing_title in seen_titles:
            similarity = SequenceMatcher(
                None, source.title.lower(), existing_title.lower()
            ).ratio()
            if similarity > 0.85:
                is_dupe_title = True
                break
        if is_dupe_title:
            continue
        seen_titles.append(source.title)

        # Limit per domain
        domain = source.domain
        current_count = domain_counts.get(domain, 0)
        if current_count >= config.max_per_domain:
            continue
        domain_counts[domain] = current_count + 1

        # Calculate relevance score
        source.relevance_score = calculate_relevance_score(source)
        unique_sources.append(source)

    # Sort by relevance score (descending)
    unique_sources.sort(key=lambda s: s.relevance_score, reverse=True)

    return unique_sources


def parse_search_results(tool_result_content: str) -> list[ResearchSource]:
    """Parse web_search tool result into ResearchSource objects.

    Args:
        tool_result_content: The content from ToolResult.

    Returns:
        List of ResearchSource objects.
    """
    sources = []

    # Parse the numbered format from web_search:
    # 1. Title
    #    URL: https://...
    #    Description...
    current: dict = {}
    lines = tool_result_content.split("\n")

    for line in lines:
        # Check for numbered title
        match = re.match(r"^\d+\.\s+(.+)$", line)
        if match:
            # Save previous if exists
            if current.get("title"):
                sources.append(
                    ResearchSource(
                        url=current.get("url", ""),
                        title=current.get("title", ""),
                        snippet=current.get("description", ""),
                    )
                )
            current = {"title": match.group(1)}
        elif line.strip().startswith("URL:"):
            current["url"] = line.strip()[4:].strip()
        elif line.strip() and "title" in current and "url" in current:
            # This is description line
            if "description" not in current:
                current["description"] = line.strip()
            else:
                current["description"] += " " + line.strip()

    # Don't forget the last one
    if current.get("title"):
        sources.append(
            ResearchSource(
                url=current.get("url", ""),
                title=current.get("title", ""),
                snippet=current.get("description", ""),
            )
        )

    return sources


@dataclass
class ResearchResult:
    """Result of research execution."""

    content: str
    sources_found: int = 0
    sources_fetched: int = 0
    queries_used: int = 0


async def execute_research(
    topic: str,
    depth: str,
    focus: str | None,
    tool_executor: "ToolExecutor",  # noqa: F821
    llm_provider: "LLMProvider",  # noqa: F821
    model: str,
    context: "ToolContext",  # noqa: F821
) -> ResearchResult:
    """Execute research workflow.

    Args:
        topic: Research topic.
        depth: Research depth (quick, standard, deep).
        focus: Optional focus area.
        tool_executor: Tool executor for web_search/web_fetch.
        llm_provider: LLM provider for query generation and synthesis.
        model: Model to use for LLM calls.
        context: Tool context.

    Returns:
        Research result with content.
    """
    from ash.llm.types import Message, Role

    config = DEPTH_CONFIGS.get(depth, DEPTH_CONFIGS["standard"])

    # Phase 1: Generate queries via LLM
    logger.info(f"Generating {config.queries} queries for: {topic}")
    query_prompt = build_query_generation_prompt(topic, config.queries, focus)

    try:
        query_response = await llm_provider.complete(
            messages=[Message(role=Role.USER, content="Generate the search queries.")],
            model=model,
            system=query_prompt,
            max_tokens=500,
        )
        queries = parse_queries_response(
            query_response.message.get_text() or "", config.queries
        )
    except Exception as e:
        logger.error(f"Failed to generate queries: {e}")
        queries = [topic]  # Fallback to just the topic

    if not queries:
        queries = [topic]

    logger.info(f"Generated {len(queries)} queries")

    # Phase 2: Execute searches in parallel
    all_sources: list[ResearchSource] = []
    search_tasks = []

    for query in queries:
        search_tasks.append(
            tool_executor.execute("web_search", {"query": query, "count": 10}, context)
        )

    search_results = await asyncio.gather(*search_tasks, return_exceptions=True)

    for result in search_results:
        if isinstance(result, Exception):
            logger.warning(f"Search failed: {result}")
            continue
        if result.is_error:
            logger.warning(f"Search error: {result.content}")
            continue
        sources = parse_search_results(result.content)
        all_sources.extend(sources)

    if not all_sources:
        return ResearchResult(
            content=f"No sources found for research topic: {topic}",
            queries_used=len(queries),
        )

    # Phase 3: Dedupe and rank
    ranked_sources = dedupe_and_rank_sources(all_sources, config)
    logger.info(
        f"Found {len(all_sources)} sources, "
        f"{len(ranked_sources)} after dedup, "
        f"fetching top {config.sources_to_fetch}"
    )

    # Phase 4: Fetch top sources in parallel
    sources_to_fetch = ranked_sources[: config.sources_to_fetch]
    fetch_tasks = []

    for source in sources_to_fetch:
        fetch_tasks.append(
            tool_executor.execute("web_fetch", {"url": source.url}, context)
        )

    fetch_results = await asyncio.gather(*fetch_tasks, return_exceptions=True)

    fetched_count = 0
    for i, result in enumerate(fetch_results):
        if isinstance(result, Exception):
            logger.warning(f"Fetch failed for {sources_to_fetch[i].url}: {result}")
            continue
        if result.is_error:
            logger.warning(
                f"Fetch error for {sources_to_fetch[i].url}: {result.content}"
            )
            continue
        sources_to_fetch[i].content = result.content
        fetched_count += 1

    logger.info(f"Successfully fetched {fetched_count}/{len(sources_to_fetch)} sources")

    # Phase 5: Synthesize via LLM
    synthesis_prompt = build_synthesis_prompt(topic, sources_to_fetch, focus)

    try:
        synthesis_response = await llm_provider.complete(
            messages=[Message(role=Role.USER, content="Create the research report.")],
            model=model,
            system=synthesis_prompt,
            max_tokens=4000,
        )
        report = synthesis_response.message.get_text() or ""
    except Exception as e:
        logger.error(f"Synthesis failed: {e}")
        # Fallback: return raw source summaries
        report = _build_fallback_report(topic, sources_to_fetch)

    return ResearchResult(
        content=report,
        sources_found=len(all_sources),
        sources_fetched=fetched_count,
        queries_used=len(queries),
    )


def _build_fallback_report(topic: str, sources: list[ResearchSource]) -> str:
    """Build a basic report if synthesis fails."""
    lines = [f"# Research: {topic}", "", "## Sources Found", ""]

    for i, source in enumerate(sources, 1):
        lines.append(f"### [{i}] {source.title}")
        lines.append(f"URL: {source.url}")
        lines.append("")
        if source.content:
            # Truncate content
            content = (
                source.content[:500] + "..."
                if len(source.content) > 500
                else source.content
            )
            lines.append(content)
        else:
            lines.append(source.snippet)
        lines.append("")

    lines.append("---")
    lines.append("*Note: Automated synthesis failed. Raw sources shown above.*")

    return "\n".join(lines)
