"""High-level command execution in sandbox containers."""

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from ash.sandbox.manager import SandboxConfig, SandboxManager

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of command execution."""

    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        """Check if execution was successful."""
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Get combined output (stdout + stderr)."""
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)


class SandboxExecutor:
    """Execute commands in isolated Docker containers."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        dockerfile_path: Path | None = None,
        environment: dict[str, str] | None = None,
    ):
        """Initialize executor.

        Args:
            config: Sandbox configuration.
            dockerfile_path: Path to Dockerfile for building image.
            environment: Environment variables to set in container.
        """
        self._config = config or SandboxConfig()
        self._manager = SandboxManager(self._config)
        self._dockerfile_path = dockerfile_path
        self._environment = environment or {}
        self._container_id: str | None = None
        self._initialized = False

    async def initialize(self) -> bool:
        """Initialize the executor, ensuring image exists.

        Returns:
            True if initialization successful.
        """
        if self._initialized:
            return True

        # Ensure image exists
        if not await self._manager.ensure_image(self._dockerfile_path):
            logger.error("Failed to ensure sandbox image")
            return False

        self._initialized = True
        return True

    async def execute(
        self,
        command: str,
        timeout: int | None = None,
        reuse_container: bool = True,
        environment: dict[str, str] | None = None,
    ) -> ExecutionResult:
        """Execute a command in the sandbox.

        Args:
            command: Shell command to execute.
            timeout: Execution timeout in seconds.
            reuse_container: Reuse existing container if available.
            environment: Extra environment variables for this command.

        Returns:
            Execution result.
        """
        if not self._initialized:
            if not await self.initialize():
                return ExecutionResult(
                    exit_code=-1,
                    stdout="",
                    stderr="Sandbox not initialized",
                    timed_out=False,
                )

        # Get or create container
        container_id = await self._get_or_create_container(reuse_container)

        # Merge base environment with per-command environment
        merged_env = {**self._environment, **(environment or {})}

        # Execute command
        try:
            exit_code, stdout, stderr = await self._manager.exec_command(
                container_id,
                command,
                timeout=timeout,
                environment=merged_env if merged_env else None,
            )

            timed_out = exit_code == -1 and "timed out" in stderr.lower()

            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=timed_out,
            )

        except Exception as e:
            logger.error(f"Execution failed: {e}", exc_info=True)
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr=str(e),
                timed_out=False,
            )

    async def execute_script(
        self,
        script: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        """Execute a multi-line script in the sandbox.

        Args:
            script: Script content.
            timeout: Execution timeout in seconds.

        Returns:
            Execution result.
        """
        # Escape script for bash -c
        escaped = script.replace("'", "'\\''")
        command = f"bash -c '{escaped}'"
        return await self.execute(command, timeout=timeout)

    async def write_file(
        self,
        path: str,
        content: str,
    ) -> ExecutionResult:
        """Write a file in the sandbox.

        Args:
            path: File path in sandbox.
            content: File content.

        Returns:
            Execution result.
        """
        # Quote path to prevent shell injection
        safe_path = shlex.quote(path)
        # Escape content for cat heredoc
        escaped = content.replace("'", "'\\''")
        command = f"cat > {safe_path} << 'ASHEOF'\n{escaped}\nASHEOF"
        return await self.execute(command)

    async def read_file(self, path: str) -> ExecutionResult:
        """Read a file from the sandbox.

        Args:
            path: File path in sandbox.

        Returns:
            Execution result with file content in stdout.
        """
        # Quote path to prevent shell injection
        safe_path = shlex.quote(path)
        return await self.execute(f"cat {safe_path}")

    async def cleanup(self) -> None:
        """Clean up the sandbox container."""
        if self._container_id:
            try:
                await self._manager.remove_container(self._container_id)
            except Exception as e:
                logger.warning(f"Failed to remove container: {e}")
            finally:
                self._container_id = None

    async def _get_or_create_container(self, reuse: bool) -> str:
        """Get existing container or create new one.

        Args:
            reuse: Whether to reuse existing container.

        Returns:
            Container ID.
        """
        if reuse and self._container_id:
            return self._container_id

        # Create new container with environment variables
        container_id = await self._manager.create_container(
            environment=self._environment if self._environment else None,
        )
        await self._manager.start_container(container_id)

        if reuse:
            self._container_id = container_id

        return container_id

    async def __aenter__(self) -> "SandboxExecutor":
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.cleanup()
