"""Bash command execution tool using Docker sandbox."""

from typing import Any

from ash.sandbox import SandboxConfig, SandboxExecutor
from ash.tools.base import Tool, ToolContext, ToolResult


class BashTool(Tool):
    """Execute bash commands in a sandboxed Docker container.

    This tool provides safe execution of shell commands in an isolated
    environment with resource limits and network isolation.
    """

    def __init__(
        self,
        sandbox_config: SandboxConfig | None = None,
        max_output_length: int = 10000,
    ):
        """Initialize bash tool.

        Args:
            sandbox_config: Sandbox configuration.
            max_output_length: Maximum output length to return.
        """
        self._executor = SandboxExecutor(config=sandbox_config)
        self._max_output_length = max_output_length

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute bash commands in a sandboxed Linux environment. "
            "Useful for running scripts, processing data, and system operations. "
            "The environment is isolated with no network access by default."
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Execution timeout in seconds (default: 60).",
                    "default": 60,
                },
            },
            "required": ["command"],
        }

    async def execute(
        self,
        input_data: dict[str, Any],
        context: ToolContext,
    ) -> ToolResult:
        """Execute the bash command in sandbox.

        Args:
            input_data: Must contain 'command' key.
            context: Execution context.

        Returns:
            Tool result with command output.
        """
        command = input_data.get("command")
        if not command:
            return ToolResult.error("Missing required parameter: command")

        timeout = input_data.get("timeout", 60)

        try:
            result = await self._executor.execute(
                command,
                timeout=timeout,
                reuse_container=True,
            )

            # Truncate output if too long
            output = result.output
            truncated = False
            if len(output) > self._max_output_length:
                output = output[: self._max_output_length]
                truncated = True

            if result.timed_out:
                return ToolResult.error(
                    f"Command timed out after {timeout} seconds.\n"
                    f"Partial output:\n{output}",
                    exit_code=-1,
                    timed_out=True,
                    truncated=truncated,
                )

            if result.success:
                content = output if output else "(no output)"
                return ToolResult.success(
                    content,
                    exit_code=result.exit_code,
                    truncated=truncated,
                )
            else:
                # Command failed but didn't error
                return ToolResult(
                    content=f"Exit code {result.exit_code}:\n{output}",
                    is_error=False,  # Non-zero exit is not an error, just a result
                    metadata={
                        "exit_code": result.exit_code,
                        "truncated": truncated,
                    },
                )

        except Exception as e:
            return ToolResult.error(f"Execution error: {e}")

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        await self._executor.cleanup()
