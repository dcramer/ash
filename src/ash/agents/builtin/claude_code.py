"""Claude Code CLI passthrough agent."""

import asyncio
import json
import logging
import shutil

from ash.agents.base import Agent, AgentConfig, AgentContext, AgentResult

logger = logging.getLogger(__name__)


class ClaudeCodeAgent(Agent):
    """Delegate tasks to Claude Code CLI with full permissions.

    This is a passthrough agent that invokes the `claude` CLI directly
    rather than running an LLM loop. It passes the input message as a prompt
    and returns the result.
    """

    @property
    def config(self) -> AgentConfig:
        return AgentConfig(
            name="claude-code",
            description="Delegate tasks to Claude Code CLI with full permissions",
            system_prompt="",  # Not used for passthrough agents
            is_passthrough=True,
        )

    async def execute_passthrough(
        self,
        message: str,
        context: AgentContext,
        model: str | None = None,
    ) -> AgentResult:
        """Execute the Claude CLI with the given prompt.

        Runs: claude --dangerously-skip-permissions -p "<message>" --output-format stream-json

        Args:
            message: The prompt to send to Claude CLI.
            context: Execution context (unused for this agent).
            model: Optional model to use (e.g., "sonnet", "opus", "haiku").

        Returns:
            AgentResult with the combined assistant response text.
        """
        # Check if claude CLI is available
        claude_path = shutil.which("claude")
        if not claude_path:
            return AgentResult.error(
                "Claude CLI not found. Please install it: "
                "https://docs.anthropic.com/en/docs/claude-code"
            )

        cmd = [
            claude_path,
            "--dangerously-skip-permissions",
            "-p",
            message,
            "--output-format",
            "stream-json",
        ]

        # Add model flag if specified
        if model:
            cmd.extend(["--model", model])
            logger.info(
                "claude_cli_executing",
                extra={"gen_ai.request.model": model},
            )
        else:
            logger.info("claude_cli_executing")

        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = stderr.decode("utf-8", errors="replace").strip()
                logger.error(
                    "claude_cli_failed",
                    extra={
                        "process.exit_code": process.returncode,
                        "error.message": error_msg,
                    },
                )
                return AgentResult.error(f"Claude CLI error: {error_msg}")

            # Parse the JSON stream output
            response_text = self._parse_stream_json(
                stdout.decode("utf-8", errors="replace")
            )

            if not response_text:
                return AgentResult.error("Claude CLI returned no response")

            return AgentResult.success(response_text)

        except Exception as e:
            logger.error("claude_cli_execution_failed", extra={"error.message": str(e)})
            return AgentResult.error(f"Failed to execute Claude CLI: {e}")

    def _parse_stream_json(self, output: str) -> str:
        """Parse Claude CLI stream-json output and extract assistant responses.

        The stream-json format outputs one JSON object per line. We look for
        assistant message events and combine their text content.

        Args:
            output: Raw stdout from the Claude CLI.

        Returns:
            Combined text from all assistant responses.
        """
        text_parts: list[str] = []

        for line in output.strip().split("\n"):
            if not line:
                continue

            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                logger.debug(f"Skipping non-JSON line: {line[:100]}")
                continue

            # Handle different event types from stream-json
            event_type = event.get("type")

            if event_type == "assistant":
                # Direct assistant message
                message = event.get("message", {})
                content = message.get("content", [])
                for block in content:
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

            elif event_type == "content_block_delta":
                # Streaming delta
                delta = event.get("delta", {})
                if delta.get("type") == "text_delta":
                    text_parts.append(delta.get("text", ""))

            elif event_type == "result":
                # Final result event - extract text from result
                result = event.get("result", "")
                if isinstance(result, str) and result:
                    text_parts.append(result)

        return "".join(text_parts)
