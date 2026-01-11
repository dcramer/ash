"""Docker container management for sandboxed execution."""

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

import docker

logger = logging.getLogger(__name__)

DEFAULT_IMAGE = "ash-sandbox:latest"
DEFAULT_TIMEOUT = 60
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = 1.0

# Network modes
NetworkMode = Literal["none", "bridge"]
# Workspace access levels
WorkspaceAccess = Literal["none", "ro", "rw"]
# Container runtime
Runtime = Literal["runc", "runsc"]  # runsc = gVisor


@dataclass
class SandboxConfig:
    """Configuration for sandbox containers."""

    image: str = DEFAULT_IMAGE
    timeout: int = DEFAULT_TIMEOUT
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: float = DEFAULT_CPU_LIMIT
    work_dir: str = "/workspace"

    # Container runtime: "runc" (default) or "runsc" (gVisor for enhanced security)
    runtime: Runtime = "runc"

    # Network configuration
    network_mode: NetworkMode = "none"  # "none" = isolated, "bridge" = has network
    dns_servers: list[str] = field(default_factory=list)  # Custom DNS for filtering
    http_proxy: str | None = None  # HTTP proxy URL for monitoring traffic

    # Workspace mounting
    workspace_path: Path | None = None  # Host path to mount
    workspace_access: WorkspaceAccess = "rw"  # none, ro, or rw


class SandboxManager:
    """Manage Docker containers for sandboxed code execution.

    Security features:
    - Read-only root filesystem (immutable container)
    - All capabilities dropped
    - No privilege escalation
    - Process limits (fork bomb protection)
    - Memory limits
    - Non-root user execution
    - Optional gVisor runtime for syscall isolation
    - tmpfs for writable areas with size limits
    """

    def __init__(self, config: SandboxConfig | None = None):
        """Initialize sandbox manager.

        Args:
            config: Sandbox configuration.
        """
        self._config = config or SandboxConfig()
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, Container] = {}

    @property
    def client(self) -> docker.DockerClient:
        """Get Docker client, initializing if needed."""
        if self._client is None:
            self._client = docker.from_env()
        return self._client

    async def ensure_image(self, dockerfile_path: Path | None = None) -> bool:
        """Ensure the sandbox image exists, building if necessary.

        Args:
            dockerfile_path: Path to Dockerfile.sandbox for building.

        Returns:
            True if image is available.
        """
        try:
            self.client.images.get(self._config.image)
            logger.debug(f"Image {self._config.image} found")
            return True
        except ImageNotFound:
            if dockerfile_path and dockerfile_path.exists():
                logger.info(f"Building image {self._config.image}")
                await self._build_image(dockerfile_path)
                return True
            logger.error(
                f"Image {self._config.image} not found and no Dockerfile provided"
            )
            return False

    async def _build_image(self, dockerfile_path: Path) -> None:
        """Build the sandbox image.

        Args:
            dockerfile_path: Path to Dockerfile.
        """
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None,
            lambda: self.client.images.build(
                path=str(dockerfile_path.parent),
                dockerfile=dockerfile_path.name,
                tag=self._config.image,
                rm=True,
            ),
        )

    async def create_container(
        self,
        name: str | None = None,
        environment: dict[str, str] | None = None,
        extra_volumes: dict[str, dict[str, str]] | None = None,
    ) -> str:
        """Create a new sandbox container with security hardening.

        Args:
            name: Optional container name.
            environment: Environment variables.
            extra_volumes: Additional volume mounts.

        Returns:
            Container ID.
        """
        # Build environment with proxy settings if configured
        env = dict(environment) if environment else {}
        if self._config.http_proxy:
            env.update(
                {
                    "HTTP_PROXY": self._config.http_proxy,
                    "HTTPS_PROXY": self._config.http_proxy,
                    "http_proxy": self._config.http_proxy,
                    "https_proxy": self._config.http_proxy,
                }
            )

        # Build volume mounts
        volumes = dict(extra_volumes) if extra_volumes else {}
        if (
            self._config.workspace_path
            and self._config.workspace_access != "none"
            and self._config.workspace_path.exists()
        ):
            mode = "ro" if self._config.workspace_access == "ro" else "rw"
            volumes[str(self._config.workspace_path)] = {
                "bind": self._config.work_dir,
                "mode": mode,
            }

        # Security-hardened container configuration
        container_config: dict[str, Any] = {
            "image": self._config.image,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": self._config.work_dir,
            # Resource limits
            "mem_limit": self._config.memory_limit,
            "nano_cpus": int(self._config.cpu_limit * 1e9),
            # Security hardening
            "read_only": True,  # Immutable root filesystem
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],  # Drop all capabilities
            "pids_limit": 100,  # Fork bomb protection
            # Writable areas via tmpfs (with size limits and security options)
            # uid=1000,gid=1000 matches the sandbox user created in Dockerfile
            "tmpfs": {
                "/tmp": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",  # noqa: S108
                "/home/sandbox": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",
                "/var/tmp": "size=32m,noexec,nosuid,nodev,uid=1000,gid=1000",  # noqa: S108
                "/run": "size=16m,noexec,nosuid,nodev,uid=1000,gid=1000",
            },
        }

        # Use gVisor runtime if configured (enhanced syscall isolation)
        if self._config.runtime == "runsc":
            container_config["runtime"] = "runsc"

        # Network configuration
        if self._config.network_mode == "none":
            container_config["network_disabled"] = True
        else:
            container_config["network_disabled"] = False
            container_config["network_mode"] = self._config.network_mode
            # Custom DNS for filtering/logging
            if self._config.dns_servers:
                container_config["dns"] = self._config.dns_servers

        if name:
            container_config["name"] = name

        if env:
            container_config["environment"] = env

        if volumes:
            container_config["volumes"] = volumes

        loop = asyncio.get_event_loop()
        container = await loop.run_in_executor(
            None,
            lambda: self.client.containers.create(**container_config),
        )

        self._containers[container.id] = container
        logger.debug(f"Created container {container.id[:12]}")
        return container.id

    async def start_container(self, container_id: str) -> None:
        """Start a container.

        Args:
            container_id: Container ID.
        """
        container = self._get_container(container_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, container.start)
        logger.debug(f"Started container {container_id[:12]}")

    async def stop_container(self, container_id: str, timeout: int = 10) -> None:
        """Stop a container.

        Args:
            container_id: Container ID.
            timeout: Stop timeout in seconds.
        """
        container = self._get_container(container_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: container.stop(timeout=timeout))
        logger.debug(f"Stopped container {container_id[:12]}")

    async def remove_container(self, container_id: str, force: bool = True) -> None:
        """Remove a container.

        Args:
            container_id: Container ID.
            force: Force removal even if running.
        """
        container = self._get_container(container_id)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, lambda: container.remove(force=force))
        self._containers.pop(container_id, None)
        logger.debug(f"Removed container {container_id[:12]}")

    async def exec_command(
        self,
        container_id: str,
        command: str | list[str],
        timeout: int | None = None,
        user: str = "sandbox",
        work_dir: str | None = None,
        environment: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        """Execute a command in a container.

        Args:
            container_id: Container ID.
            command: Command to execute.
            timeout: Execution timeout (uses config default if None).
            user: User to run command as.
            work_dir: Working directory for command.
            environment: Environment variables for this command.

        Returns:
            Tuple of (exit_code, stdout, stderr).
        """
        container = self._get_container(container_id)
        timeout = timeout or self._config.timeout

        exec_config = {
            "cmd": command
            if isinstance(command, list)
            else ["/bin/bash", "-c", command],
            "user": user,
            "tty": False,
            "stdout": True,
            "stderr": True,
        }

        if work_dir:
            exec_config["workdir"] = work_dir

        if environment:
            exec_config["environment"] = [f"{k}={v}" for k, v in environment.items()]

        loop = asyncio.get_event_loop()

        # Create exec instance
        exec_instance = await loop.run_in_executor(
            None,
            lambda: self.client.api.exec_create(container.id, **exec_config),
        )

        # Start exec and get output with timeout
        try:
            output = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self.client.api.exec_start(
                        exec_instance["Id"],
                        demux=True,
                    ),
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(f"Command timed out after {timeout}s")
            return -1, "", f"Command timed out after {timeout} seconds"

        # Get exit code
        inspect_result = await loop.run_in_executor(
            None,
            lambda: self.client.api.exec_inspect(exec_instance["Id"]),
        )
        exit_code = inspect_result.get("ExitCode", -1)

        # Decode output, handling binary content gracefully
        stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
        stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""

        return exit_code, stdout, stderr

    async def cleanup_all(self) -> None:
        """Stop and remove all managed containers."""
        for container_id in list(self._containers.keys()):
            try:
                await self.remove_container(container_id, force=True)
            except NotFound:
                self._containers.pop(container_id, None)

    def _get_container(self, container_id: str) -> Container:
        """Get a container by ID.

        Args:
            container_id: Container ID.

        Returns:
            Container instance.

        Raises:
            KeyError: If container not found.
        """
        if container_id not in self._containers:
            # Try to get from Docker
            try:
                container = self.client.containers.get(container_id)
                self._containers[container_id] = container
            except NotFound as e:
                raise KeyError(f"Container {container_id} not found") from e
        return self._containers[container_id]

    def __del__(self):
        """Clean up on destruction."""
        if self._client:
            try:
                self._client.close()
            except Exception:
                # Ignore errors during cleanup - client may already be closed
                logger.debug("Error closing Docker client during cleanup")
