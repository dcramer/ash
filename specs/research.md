# Research

> Deep research subagent with multi-query orchestration and source synthesis

Files: src/ash/skills/research.py, src/ash/skills/executor.py

## Requirements

### MUST

- Be invoked as a dynamic skill via SkillExecutor (like write-skill)
- Generate diverse search queries covering different angles
- Execute web_search for each query
- Deduplicate sources by URL (exact match)
- Deduplicate sources by title similarity (fuzzy match >85%)
- Limit sources per domain (max 3)
- Fetch content from top sources via web_fetch
- Handle fetch failures gracefully (continue with available sources)
- Synthesize findings via LLM with citation instructions
- Produce inline citations [1], [2], [3] throughout report
- Include numbered source list at end with titles and URLs
- Support three depth levels: quick, standard, deep

### SHOULD

- Execute searches in parallel (asyncio.gather)
- Execute fetches in parallel
- Rank sources by domain authority (.edu, .gov higher)
- Track and report methodology (queries used, sources found vs fetched)
- Note conflicting information between sources
- Include executive summary at report start

### MAY

- Support focus parameter to guide query generation
- Cache research results by topic hash
- Include confidence indicators for findings
- Detect and flag outdated sources

## Interface

```python
# Invoked via SkillExecutor
skill_executor.execute(
    "research",
    {
        "topic": "How do modern AI agents handle web search?",
        "depth": "standard",  # optional, default: standard
        "focus": "architecture",  # optional
    },
    context,
)

@dataclass
class ResearchSource:
    url: str
    title: str
    snippet: str
    content: str | None = None
    domain: str = ""
    relevance_score: float = 0.0

@dataclass
class ResearchConfig:
    queries: int
    sources_to_fetch: int
    max_per_domain: int = 3

@dataclass
class ResearchResult:
    content: str
    sources_found: int = 0
    sources_fetched: int = 0
    queries_used: int = 0

async def execute_research(
    topic: str,
    depth: str,
    focus: str | None,
    tool_executor: ToolExecutor,
    llm_provider: LLMProvider,
    model: str,
    context: ToolContext,
) -> ResearchResult: ...
```

## Depth Levels

| Depth | Queries | Sources Fetched | Description |
|-------|---------|-----------------|-------------|
| quick | 2 | 3 | Fast, surface-level |
| standard | 5 | 10 | Balanced depth |
| deep | 10 | 20 | Comprehensive |

## Workflow Phases

1. **Query Generation**: LLM generates N diverse queries from topic
2. **Search Execution**: Parallel web_search for each query
3. **Dedup & Ranking**: Programmatic - URL dedup, title similarity, domain limits
4. **Content Fetching**: Parallel web_fetch for top M sources
5. **Synthesis**: LLM produces cited report from fetched content

## Output Format

```markdown
# Research: {topic}

## Summary
[2-3 sentence executive summary of key findings]

## Findings

### {Subtopic 1}
[Findings with inline citations [1][2]]

### {Subtopic 2}
[More findings [3][4]]

## Methodology
- Queries executed: N
- Sources found: X
- Sources analyzed: Y
- Depth: {depth}

## Sources
[1] Title One - https://example.com/article1
[2] Title Two - https://example.org/article2
...
```

## Errors

| Condition | Response |
|-----------|----------|
| Missing topic | SkillResult.error("Missing required input: topic") |
| Invalid depth | SkillResult.error("Invalid depth: must be quick, standard, or deep") |
| All searches failed | SkillResult.error("All search queries failed") |
| All fetches failed | Continue with snippets only, note in report |
| web_search unavailable | SkillResult.error("Research requires web_search tool") |

## Verification

```bash
uv run pytest tests/test_research.py -v
```

- Research skill registered in SkillExecutor
- Queries generated from topic
- Sources deduplicated by URL and title
- Domain limits enforced
- Synthesis produces cited report
- Parallel execution for searches and fetches
