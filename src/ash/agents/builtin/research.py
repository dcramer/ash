"""Research agent for deep web research with synthesis."""

from ash.agents.base import Agent, AgentConfig, AgentContext

RESEARCH_SYSTEM_PROMPT = """You are a research assistant. Research the given topic thoroughly.

## Process

1. **Search broadly** - Use 4-6 different search queries to explore the topic from multiple angles
2. **Read authoritative sources** - Fetch content from official documentation, .gov, .edu sites
3. **Synthesize findings** - Combine information with inline citations [1], [2]

## Output Format

Provide a structured response with:

- **Summary**: 2-3 sentence executive summary
- **Findings**: Detailed analysis with inline citations
- **Sources**: Numbered list of URLs used

## Best Practices

- Prefer official documentation over blog posts
- Cross-reference information across multiple sources
- Note when sources disagree
- Include publication dates when available
"""


class ResearchAgent(Agent):
    """Deep research with web search and synthesis."""

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="research",
            description="Research a topic using web search to find authoritative sources",
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            allowed_tools=["web_search", "web_fetch"],
            max_iterations=15,
        )

    def build_system_prompt(self, context: AgentContext) -> str:
        prompt = self.config.system_prompt
        focus = context.input_data.get("focus")
        if focus:
            prompt += f"\n\n## Focus Area\n\nPay special attention to: {focus}"
        return prompt
