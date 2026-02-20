"""Tool result summarization using a fast/cheap model.

When tool outputs are large but not large enough to truncate,
summarization can help preserve context window while retaining key information.
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from ash.tools.truncation import _save_to_temp

if TYPE_CHECKING:
    from ash.config import AshConfig
    from ash.llm import LLMProvider

logger = logging.getLogger(__name__)

# Default threshold for summarization (2KB)
SUMMARIZE_THRESHOLD_BYTES = 2 * 1024

# System prompt for summarization
SUMMARIZE_SYSTEM_PROMPT = """You are a technical summarization assistant. Your task is to create concise summaries of command output or file content.

Guidelines:
- Extract the key information that would be most relevant to someone debugging or developing
- Preserve important details like error messages, status codes, file paths, and specific values
- Use bullet points for multiple distinct pieces of information
- Keep the summary under 500 words
- If the content appears to be an error or failure, highlight that prominently
- For code output, note the structure and key functions/classes
- For command output, note success/failure status and key results"""

SUMMARIZE_USER_PROMPT = """Summarize the following {content_type} output. Focus on the most important information:

---
{content}
---

Provide a concise summary that captures the key points."""


@dataclass
class SummarizationResult:
    """Result of summarization operation."""

    content: str
    summarized: bool
    original_bytes: int
    summary_bytes: int
    full_output_path: str | None = None
    error: str | None = None

    def to_metadata(self) -> dict:
        """Convert to metadata dict for ToolResult.

        Note: full_output_path is intentionally excluded from agent-facing
        metadata since it's a host path the agent cannot access.
        """
        meta = {
            "summarized": self.summarized,
            "original_bytes": self.original_bytes,
        }
        if self.summarized:
            meta["summary_bytes"] = self.summary_bytes
            # full_output_path excluded - host path not accessible to agent
        if self.error:
            meta["summarization_error"] = self.error
        return meta


@dataclass
class ToolResultSummarizer:
    """Summarizes large tool outputs using a fast LLM.

    Example usage:
        summarizer = ToolResultSummarizer(llm_provider, model="gpt-5-mini")

        # In tool executor or agent:
        result = await tool.execute(input_data, context)
        if summarizer:
            result = await summarizer.maybe_summarize(result, tool_name="bash")
    """

    llm: "LLMProvider"
    model: str
    threshold_bytes: int = SUMMARIZE_THRESHOLD_BYTES
    max_summary_tokens: int = 500
    enabled: bool = True
    _stats: dict = field(default_factory=lambda: {"calls": 0, "bytes_saved": 0})

    async def maybe_summarize(
        self,
        content: str,
        content_type: str = "command",
        save_full: bool = True,
    ) -> SummarizationResult:
        """Summarize content if it exceeds the threshold.

        Args:
            content: The content to potentially summarize.
            content_type: Type of content for prompt (e.g., "command", "file").
            save_full: Whether to save full content to temp file.

        Returns:
            SummarizationResult with either original or summarized content.
        """
        original_bytes = len(content.encode("utf-8"))

        # Check if summarization is needed
        if not self.enabled or original_bytes <= self.threshold_bytes:
            return SummarizationResult(
                content=content,
                summarized=False,
                original_bytes=original_bytes,
                summary_bytes=original_bytes,
            )

        # Save full output to temp file first
        full_path: str | None = None
        if save_full:
            full_path = _save_to_temp(content, prefix="full_output")

        try:
            summary = await self._generate_summary(content, content_type)
            summary_bytes = len(summary.encode("utf-8"))

            # Note: don't expose host temp path to agent
            summary += f"\n\n[Summarized from {original_bytes:,} bytes]"
            summary_bytes = len(summary.encode("utf-8"))

            self._stats["calls"] += 1
            self._stats["bytes_saved"] += original_bytes - summary_bytes

            logger.debug(
                f"Summarized {original_bytes:,} bytes to {summary_bytes:,} bytes "
                f"({100 * summary_bytes / original_bytes:.1f}%)"
            )

            return SummarizationResult(
                content=summary,
                summarized=True,
                original_bytes=original_bytes,
                summary_bytes=summary_bytes,
                full_output_path=full_path,
            )

        except Exception as e:
            logger.warning("summarization_failed", extra={"error.message": str(e)})
            # Fall back to original content
            return SummarizationResult(
                content=content,
                summarized=False,
                original_bytes=original_bytes,
                summary_bytes=original_bytes,
                full_output_path=full_path,
                error=str(e),
            )

    async def _generate_summary(self, content: str, content_type: str) -> str:
        """Generate summary using the LLM.

        Args:
            content: Content to summarize.
            content_type: Type of content for prompt context.

        Returns:
            Generated summary text.
        """
        from ash.llm.types import Message, Role

        user_prompt = SUMMARIZE_USER_PROMPT.format(
            content_type=content_type,
            content=content,
        )

        response = await self.llm.complete(
            messages=[Message(role=Role.USER, content=user_prompt)],
            model=self.model,
            system=SUMMARIZE_SYSTEM_PROMPT,
            max_tokens=self.max_summary_tokens,
            temperature=0.3,  # Lower temperature for factual summarization
        )

        return response.message.get_text() or content

    @property
    def stats(self) -> dict:
        """Get summarization statistics."""
        return dict(self._stats)

    def reset_stats(self) -> None:
        """Reset summarization statistics."""
        self._stats = {"calls": 0, "bytes_saved": 0}


def create_summarizer_from_config(
    config: "AshConfig",  # noqa: F821
    model_alias: str = "default",
    threshold_bytes: int = SUMMARIZE_THRESHOLD_BYTES,
) -> ToolResultSummarizer | None:
    """Create a summarizer from application config.

    Args:
        config: Application configuration.
        model_alias: Model alias to use for summarization.
        threshold_bytes: Size threshold for summarization.

    Returns:
        Configured summarizer, or None if summarization is disabled or
        the required model/API key is not configured.
    """
    try:
        model_config = config.get_model(model_alias)
        llm = config.create_llm_provider_for_model(model_alias)

        return ToolResultSummarizer(
            llm=llm,
            model=model_config.model,
            threshold_bytes=threshold_bytes,
        )

    except Exception as e:
        logger.debug(f"Failed to create summarizer: {e}")
        return None
