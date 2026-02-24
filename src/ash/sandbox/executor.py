"""High-level command execution in sandbox containers."""

import hashlib
import json
import logging
import shlex
from dataclasses import dataclass
from pathlib import Path

from ash.config.paths import get_ash_home, get_run_path
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
                state = self._read_managed_container_state()
                if state and state.get("container_id") == self._container_id:
                    self._clear_managed_container_state()
                self._container_id = None

    async def _get_or_create_container(self, reuse: bool) -> str:
        if reuse and self._container_id:
            return self._container_id

        managed_name = self._managed_container_name() if reuse else None
        if reuse:
            reused_id = await self._resolve_managed_container(managed_name)
            if reused_id:
                self._container_id = reused_id
                return reused_id

        try:
            container_id = await self._manager.create_container(
                name=managed_name,
                environment=self._environment if self._environment else None,
            )
        except Exception:
            # Name may already exist from another executor/process racing creation.
            if reuse:
                reused_id = await self._resolve_managed_container(managed_name)
                if reused_id:
                    self._container_id = reused_id
                    return reused_id
            raise
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
            self._write_managed_container_state(container_id, managed_name)

        return container_id

    def _managed_container_name(self) -> str:
        home = str(get_ash_home())
        suffix = hashlib.sha256(home.encode("utf-8")).hexdigest()[:8]
        return f"ash-sandbox-{suffix}"

    def _managed_state_path(self) -> Path:
        return get_run_path() / "sandbox_container.json"

    def _read_managed_container_state(self) -> dict[str, str] | None:
        path = self._managed_state_path()
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        container_id = payload.get("container_id")
        container_name = payload.get("container_name")
        if not isinstance(container_id, str) or not isinstance(container_name, str):
            return None
        return {"container_id": container_id, "container_name": container_name}

    def _write_managed_container_state(
        self, container_id: str, container_name: str | None
    ) -> None:
        if not container_name:
            return
        path = self._managed_state_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"container_id": container_id, "container_name": container_name}
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2))

    def _clear_managed_container_state(self) -> None:
        path = self._managed_state_path()
        if path.exists():
            path.unlink()

    async def _resolve_managed_container(self, managed_name: str | None) -> str | None:
        if not managed_name:
            return None
        expected_image_id = await self._manager.get_image_id(self._config.image)
        state = self._read_managed_container_state()
        refs: list[str] = []
        if state:
            refs.extend([state["container_id"], state["container_name"]])
        refs.append(managed_name)

        seen: set[str] = set()
        for ref in refs:
            if not ref or ref in seen:
                continue
            seen.add(ref)
            container = await self._manager.get_container(ref)
            if container is None:
                continue
            attrs = container.attrs if isinstance(container.attrs, dict) else {}
            container_image_id = str(attrs.get("Image") or "")
            if (
                expected_image_id
                and container_image_id
                and container_image_id != expected_image_id
            ):
                logger.info(
                    "sandbox_container_image_id_mismatch_pruned",
                    extra={
                        "container.id": container.id[:12],
                        "container.image_id": container_image_id,
                        "sandbox.image": self._config.image,
                        "sandbox.image_id": expected_image_id,
                    },
                )
                await self._manager.remove_container(container.id, force=True)
                continue
            config = attrs.get("Config", {})
            config_image = (
                str(config.get("Image") or "") if isinstance(config, dict) else ""
            )
            # Fallback for older container metadata: tag/name should match config image.
            # TODO(2026-03-10): Remove this fallback after stale managed containers
            # have rolled out and image-id matching has been in production for a bit.
            if config_image and config_image not in {
                self._config.image,
                expected_image_id,
            }:
                logger.info(
                    "sandbox_container_image_mismatch_pruned",
                    extra={
                        "container.id": container.id[:12],
                        "container.image": config_image,
                        "sandbox.image": self._config.image,
                    },
                )
                await self._manager.remove_container(container.id, force=True)
                continue
            if not self._container_has_expected_rpc_mount(container):
                logger.info(
                    "sandbox_container_rpc_mount_mismatch_pruned",
                    extra={
                        "container.id": container.id[:12],
                        "rpc.socket": str(self._config.rpc_socket_path),
                    },
                )
                await self._manager.remove_container(container.id, force=True)
                continue
            status = await self._manager.get_container_status(container.id)
            if status != "running":
                await self._manager.start_container(container.id)
            self._write_managed_container_state(container.id, managed_name)
            return container.id

        self._clear_managed_container_state()
        return None

    def _container_has_expected_rpc_mount(self, container: object) -> bool:
        rpc_socket_path = self._config.rpc_socket_path
        if rpc_socket_path is None:
            return True

        expected_source = str(rpc_socket_path.parent)
        expected_dest = f"{self._config.mount_prefix}/run"

        attrs = getattr(container, "attrs", None)
        if not isinstance(attrs, dict):
            return False
        mounts = attrs.get("Mounts")
        if not isinstance(mounts, list):
            return False

        for mount in mounts:
            if not isinstance(mount, dict):
                continue
            if (
                str(mount.get("Source", "")) == expected_source
                and str(mount.get("Destination", "")) == expected_dest
            ):
                return True
        return False

    async def __aenter__(self) -> "SandboxExecutor":
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.cleanup()
