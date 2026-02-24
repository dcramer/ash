"""High-level command execution in sandbox containers."""

import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from ash.sandbox.manager import SandboxConfig, SandboxManager

logger = logging.getLogger(__name__)


def _normalize_workspace_path(path: str) -> str:
    original = path
    while path.startswith("/workspace/workspace"):
        path = path.replace("/workspace/workspace", "/workspace", 1)
    while "//" in path:
        path = path.replace("//", "/")
    path = path.rstrip("/")
    if path != original:
        logger.debug(f"Normalized path: {original} -> {path}")
    return path


@dataclass
class ExecutionResult:
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        parts = [p for p in (self.stdout, self.stderr) if p]
        return "\n".join(parts)


class SandboxExecutor:
    def __init__(
        self,
        config: SandboxConfig | None = None,
        dockerfile_path: Path | None = None,
        environment: dict[str, str] | None = None,
        setup_command: str | None = None,
    ):
        self._config = config or SandboxConfig()
        self._manager = SandboxManager(self._config)
        self._dockerfile_path = dockerfile_path
        self._environment = environment or {}
        self._setup_command = setup_command
        self._container_id: str | None = None
        self._container_setup_done: bool = False
        self._initialized = False

    async def initialize(self) -> bool:
        if self._initialized:
            return True
        if not await self._manager.ensure_image(self._dockerfile_path):
            logger.error("sandbox_image_ensure_failed")
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
        if not self._initialized and not await self.initialize():
            return ExecutionResult(
                exit_code=-1,
                stdout="",
                stderr="Sandbox not initialized",
            )

        try:
            container_id = await self._get_or_create_container(reuse_container)
        except Exception as e:
            logger.error(
                "sandbox_execution_failed",
                extra={"error.message": str(e)},
                exc_info=True,
            )
            return ExecutionResult(exit_code=-1, stdout="", stderr=str(e))
        ephemeral_container = not reuse_container
        merged_env = {**self._environment, **(environment or {})}

        try:
            exit_code, stdout, stderr = await self._manager.exec_command(
                container_id,
                command,
                timeout=timeout,
                environment=merged_env if merged_env else None,
            )
            return ExecutionResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                timed_out=exit_code == -1 and "timed out" in stderr.lower(),
            )
        except Exception as e:
            logger.error(
                "sandbox_execution_failed",
                extra={"error.message": str(e)},
                exc_info=True,
            )
            return ExecutionResult(exit_code=-1, stdout="", stderr=str(e))
        finally:
            if ephemeral_container:
                try:
                    await self._manager.remove_container(container_id)
                except Exception as e:
                    logger.warning(
                        "container_removal_failed",
                        extra={
                            "error.message": str(e),
                            "container.id": container_id[:12],
                        },
                    )

    async def execute_script(
        self,
        script: str,
        timeout: int | None = None,
    ) -> ExecutionResult:
        escaped = script.replace("'", "'\\''")
        return await self.execute(f"bash -c '{escaped}'", timeout=timeout)

    async def write_file(self, path: str, content: str) -> ExecutionResult:
        import base64

        normalized_path = _normalize_workspace_path(path)
        safe_path = shlex.quote(normalized_path)
        encoded = base64.b64encode(content.encode()).decode()
        command = (
            f'mkdir -p "$(dirname {safe_path})" && '
            f"echo {shlex.quote(encoded)} | base64 -d > {safe_path}"
        )
        return await self.execute(command)

    async def read_file(self, path: str) -> ExecutionResult:
        return await self.execute(f"cat {shlex.quote(path)}")

    async def cleanup(self) -> None:
        if self._container_id:
            try:
                await self._manager.remove_container(self._container_id)
            except Exception as e:
                logger.warning(
                    "container_removal_failed", extra={"error.message": str(e)}
                )
            finally:
                self._container_id = None

    async def _get_or_create_container(self, reuse: bool) -> str:
        if reuse and self._container_id:
            return self._container_id

        container_id = await self._manager.create_container(
            environment=self._environment if self._environment else None,
        )
        try:
            await self._manager.start_container(container_id)

            if self._setup_command and not self._container_setup_done:
                logger.info("container_setup_running")
                exit_code, stdout, stderr = await self._manager.exec_command(
                    container_id,
                    self._setup_command,
                    timeout=300,
                )
                if exit_code != 0:
                    logger.warning(
                        "setup_command_failed",
                        extra={"process.exit_code": exit_code, "error.message": stderr},
                    )
                else:
                    logger.debug(
                        f"Setup command completed: {stdout[:200] if stdout else ''}"
                    )
                self._container_setup_done = True
        except Exception:
            try:
                await self._manager.remove_container(container_id)
            except Exception as remove_error:
                logger.warning(
                    "container_removal_failed",
                    extra={
                        "error.message": str(remove_error),
                        "container.id": container_id[:12],
                    },
                )
            raise

        if reuse:
            self._container_id = container_id

        return container_id

    async def __aenter__(self) -> "SandboxExecutor":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
