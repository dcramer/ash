"""Bash command execution tool with mandatory Docker sandbox."""

import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ash.sandbox import SandboxExecutor
from ash.sandbox.manager import SandboxConfig as SandboxManagerConfig
from ash.tools.base import Tool, ToolContext, ToolResult

if TYPE_CHECKING:
    from ash.config.models import SandboxConfig


class BashTool(Tool):
    """Execute bash commands in a secure Docker sandbox.

    All commands run in an isolated container with security hardening:
    - Read-only root filesystem
    - All capabilities dropped
    - No privilege escalation
    - Process limits (fork bomb protection)
    - Memory limits
    - Non-root user execution
    - Optional gVisor runtime for enhanced syscall isolation
    """

    def __init__(
        self,
        sandbox_config: "SandboxConfig | None" = None,
        workspace_path: Path | None = None,
        max_output_length: int = 10000,
    ):
        """Initialize bash tool.

        Args:
            sandbox_config: Sandbox configuration (pydantic model from config).
            workspace_path: Path to workspace to mount in sandbox.
            max_output_length: Maximum output length to return.
        """
        self._max_output_length = max_output_length
        manager_config = self._build_manager_config(sandbox_config, workspace_path)
        self._executor = SandboxExecutor(config=manager_config)

    def _build_manager_config(
        self,
        config: "SandboxConfig | None",
        workspace_path: Path | None,
    ) -> SandboxManagerConfig:
        """Convert pydantic SandboxConfig to manager's dataclass config."""
        if config is None:
            return SandboxManagerConfig(workspace_path=workspace_path)

        return SandboxManagerConfig(
            image=config.image,
            timeout=config.timeout,
            memory_limit=config.memory_limit,
            cpu_limit=config.cpu_limit,
            runtime=config.runtime,
            network_mode=config.network_mode,
            dns_servers=list(config.dns_servers) if config.dns_servers else [],
            http_proxy=config.http_proxy,
            workspace_path=workspace_path,
            workspace_access=config.workspace_access,
        )

    @property
    def name(self) -> str:
        return "bash"

    @property
    def description(self) -> str:
        return (
            "Execute bash commands in a secure sandboxed environment. "
            "Useful for running scripts, processing data, and system operations. "
            "The environment is isolated with resource limits and security hardening."
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
        """Execute the bash command in the sandbox.

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
            return await self._execute_sandboxed(command, timeout)
        except Exception as e:
            return ToolResult.error(f"Execution error: {e}")

    async def _execute_sandboxed(self, command: str, timeout: int) -> ToolResult:
        """Execute command in Docker sandbox."""
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

    async def cleanup(self) -> None:
        """Clean up sandbox resources."""
        if self._executor:
            await self._executor.cleanup()
