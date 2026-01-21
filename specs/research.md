# Research Agent

> Built-in agent for deep web research with multi-source synthesis

Files: src/ash/agents/builtin/research.py

## Requirements

### MUST

- Be invocable via `use_agent` tool with agent="research"
- Use web_search tool for querying multiple search angles
- Use web_fetch tool for reading authoritative sources
- Produce inline citations [1], [2], [3] throughout response
- Include numbered source list at end with URLs
- Provide a summary section with key findings
- Limit tool iterations to prevent runaway searches (default 15)
- Restrict tools to only web_search and web_fetch

### SHOULD

- Prefer official documentation over blog posts
- Cross-reference information across multiple sources
- Note when sources disagree
- Include publication dates when available
- Support focus parameter to guide research direction

### MAY

- Support depth parameter for controlling research thoroughness
- Track and report methodology (queries used, sources analyzed)
- Cache research results

## Interface

```python
# Invoked via use_agent tool
use_agent(
    agent="research",
    message="Research modern AI agent architectures",
    input={"focus": "tool use patterns"}  # optional
)

class ResearchAgent(Agent):
    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="research",
            description="Research a topic using web search",
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            tools=["web_search", "web_fetch"],
            max_iterations=15,
        )

    def build_system_prompt(self, context: AgentContext) -> str:
        """Build prompt with optional focus area."""
```

## Configuration

```python
AgentConfig(
    name="research",
    description="Research a topic using web search to find authoritative sources",
    tools=["web_search", "web_fetch"],
    max_iterations=15,
)
```

## Output Format

The agent produces markdown output:

```markdown
## Summary
[2-3 sentence executive summary of key findings]

## Findings
[Detailed analysis with inline citations [1][2]]

## Sources
[1] Title - https://example.com/article1
[2] Title - https://example.org/article2
```

## Behaviors

| Input | Output | Notes |
|-------|--------|-------|
| Message only | Research report | Agent chooses search strategy |
| Message + focus | Focused research | Focus guides query generation |
| Topic with few sources | Report with available sources | Agent adapts to what's found |

## Errors

| Condition | Response |
|-----------|----------|
| web_search unavailable | Error: tool not in allowed list |
| All searches fail | Agent reports inability to find sources |
| Max iterations reached | Returns partial findings |

## Design Notes

The research agent uses an agentic approach where the LLM decides:
- Which queries to run
- Which sources to fetch
- How to synthesize findings

This is simpler than a multi-phase orchestrator but relies on LLM reasoning
to produce quality research. The system prompt guides best practices.

## Verification

```bash
uv run ash chat "Use the research agent to find info about Python async"
```

- Agent invoked via use_agent tool
- Multiple web_search queries executed
- Sources fetched and cited
- Coherent report produced
