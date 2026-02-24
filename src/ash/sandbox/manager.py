"""Docker container management for sandboxed execution."""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from docker.errors import ImageNotFound, NotFound
from docker.models.containers import Container

import docker

logger = logging.getLogger(__name__)


async def _get_docker_host_async() -> str | None:
    """Get the Docker host URL, respecting the current Docker context."""
    if os.environ.get("DOCKER_HOST"):
        return None

    try:
        proc = await asyncio.create_subprocess_exec(
            "docker",
            "context",
            "inspect",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0:
            context = json.loads(stdout.decode())
            return context[0].get("Endpoints", {}).get("docker", {}).get("Host")
    except (TimeoutError, json.JSONDecodeError, FileNotFoundError, OSError):
        pass

    return None


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

    # Sessions mounting (for agent to read chat history)
    sessions_path: Path | None = None  # Host path to sessions directory
    sessions_access: Literal["none", "ro"] = "ro"  # none or ro (never rw)

    # Chats mounting (for agent to read chat state/participants)
    chats_path: Path | None = None  # Host path to chats directory

    # Schedule file mounting (for schedule.jsonl)
    schedule_file: Path | None = None  # Host path to schedule.jsonl

    # Logs mounting (for agent to inspect server logs)
    logs_path: Path | None = None  # Host path to logs directory

    # RPC runtime mounting (for sandbox to communicate with host over Unix socket)
    rpc_socket_path: Path | None = (
        None  # Host path to RPC socket (inside mounted run dir)
    )

    # UV cache mounting (for persistent package cache across sandbox runs)
    uv_cache_path: Path | None = None  # Host path to uv cache directory

    # Source code mounting (for self-debugging skills)
    source_path: Path | None = None  # Host path to Ash source code
    source_access: Literal["none", "ro"] = "none"  # Never rw - always read-only

    # Bundled skills directory (read-only, for bundled skills with references)
    bundled_skills_path: Path | None = None  # Host path to bundled skills dir

    # Mount prefix for sandbox paths (sessions, chats, logs, etc.)
    mount_prefix: str = "/ash"


class SandboxManager:
    """Manage Docker containers for sandboxed code execution."""

    def __init__(self, config: SandboxConfig | None = None):
        self._config = config or SandboxConfig()
        self._client: docker.DockerClient | None = None
        self._containers: dict[str, Container] = {}

    @property
    def client(self) -> docker.DockerClient:
        if self._client is None:
            raise RuntimeError(
                "Docker client not initialized. Call _ensure_client() first."
            )
        return self._client

    async def _ensure_client(self) -> docker.DockerClient:
        if self._client is None:
            docker_host = await _get_docker_host_async()
            self._client = (
                docker.DockerClient(base_url=docker_host)
                if docker_host
                else docker.from_env()
            )
        return self._client

    async def ensure_image(self, dockerfile_path: Path | None = None) -> bool:
        client = await self._ensure_client()
        try:
            client.images.get(self._config.image)
            logger.debug(f"Image {self._config.image} found")
            return True
        except ImageNotFound:
            if dockerfile_path and dockerfile_path.exists():
                logger.info(
                    "sandbox_image_building",
                    extra={"sandbox.image": self._config.image},
                )
                await self._build_image(dockerfile_path)
                return True
            logger.error(
                "sandbox_image_not_found",
                extra={"sandbox.image": self._config.image},
            )
            return False

    async def _build_image(self, dockerfile_path: Path) -> None:
        client = await self._ensure_client()
        await asyncio.to_thread(
            client.images.build,
            path=str(dockerfile_path.parent),
            dockerfile=dockerfile_path.name,
            tag=self._config.image,
            rm=True,
        )

    async def create_container(
        self,
        name: str | None = None,
        environment: dict[str, str] | None = None,
        extra_volumes: dict[str, dict[str, str]] | None = None,
    ) -> str:
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

        volumes = dict(extra_volumes) if extra_volumes else {}

        if (
            self._config.workspace_path
            and self._config.workspace_access != "none"
            and self._config.workspace_path.exists()
        ):
            volumes[str(self._config.workspace_path)] = {
                "bind": self._config.work_dir,
                "mode": "ro" if self._config.workspace_access == "ro" else "rw",
            }

        prefix = self._config.mount_prefix

        if (
            self._config.sessions_path
            and self._config.sessions_access != "none"
            and self._config.sessions_path.exists()
        ):
            volumes[str(self._config.sessions_path)] = {
                "bind": f"{prefix}/sessions",
                "mode": "ro",
            }

        if self._config.chats_path and self._config.chats_path.exists():
            volumes[str(self._config.chats_path)] = {
                "bind": f"{prefix}/chats",
                "mode": "ro",
            }

        if self._config.schedule_file:
            # Create schedule file if it doesn't exist
            self._config.schedule_file.parent.mkdir(parents=True, exist_ok=True)
            self._config.schedule_file.touch(exist_ok=True)
            volumes[str(self._config.schedule_file)] = {
                "bind": f"{prefix}/schedule.jsonl",
                "mode": "rw",
            }

        if self._config.logs_path and self._config.logs_path.exists():
            volumes[str(self._config.logs_path)] = {
                "bind": f"{prefix}/logs",
                "mode": "ro",
            }

        if self._config.rpc_socket_path:
            # Mount the whole run directory instead of a single socket file.
            # Socket files are frequently unlinked/recreated on server restart;
            # bind-mounting the parent dir keeps the recreated inode visible.
            run_dir = self._config.rpc_socket_path.parent
            socket_name = self._config.rpc_socket_path.name
            if run_dir.exists():
                volumes[str(run_dir)] = {
                    "bind": f"{prefix}/run",
                    "mode": "rw",
                }
                env["ASH_RPC_SOCKET"] = f"{prefix}/run/{socket_name}"

        if self._config.uv_cache_path:
            self._config.uv_cache_path.mkdir(parents=True, exist_ok=True)
            volumes[str(self._config.uv_cache_path)] = {
                "bind": "/home/sandbox/.cache/uv",
                "mode": "rw",
            }

        if (
            self._config.source_path
            and self._config.source_access != "none"
            and self._config.source_path.exists()
        ):
            volumes[str(self._config.source_path)] = {
                "bind": f"{prefix}/source",
                "mode": "ro",  # Always read-only
            }

        if (
            self._config.bundled_skills_path
            and self._config.bundled_skills_path.exists()
        ):
            volumes[str(self._config.bundled_skills_path)] = {
                "bind": f"{prefix}/skills",
                "mode": "ro",
            }

        env["ASH_MOUNT_PREFIX"] = prefix

        container_config: dict[str, Any] = {
            "image": self._config.image,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": self._config.work_dir,
            "mem_limit": self._config.memory_limit,
            "nano_cpus": int(self._config.cpu_limit * 1e9),
            "read_only": True,
            "security_opt": ["no-new-privileges:true"],
            "cap_drop": ["ALL"],
            "pids_limit": 100,
            "tmpfs": {
                "/tmp": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",  # noqa: S108
                "/home/sandbox": "size=64m,noexec,nosuid,nodev,uid=1000,gid=1000",
                "/var/tmp": "size=32m,noexec,nosuid,nodev,uid=1000,gid=1000",  # noqa: S108
                "/run": "size=16m,noexec,nosuid,nodev,uid=1000,gid=1000",
            },
            "labels": {
                "ash.managed": "true",
                "ash.component": "sandbox",
            },
        }

        if self._config.runtime == "runsc":
            container_config["runtime"] = "runsc"

        if self._config.network_mode == "none":
            container_config["network_disabled"] = True
        else:
            container_config["network_disabled"] = False
            container_config["network_mode"] = self._config.network_mode
            if self._config.dns_servers:
                container_config["dns"] = self._config.dns_servers

        if name:
            container_config["name"] = name
        if env:
            container_config["environment"] = env
        if volumes:
            container_config["volumes"] = volumes

        client = await self._ensure_client()
        container = await asyncio.to_thread(
            client.containers.create, **container_config
        )

        self._containers[container.id] = container
        logger.debug(f"Created container {container.id[:12]}")
        return container.id

    async def get_container(self, container_ref: str) -> Container | None:
        """Fetch container by id or name; return None when missing."""
        await self._ensure_client()
        try:
            return await asyncio.to_thread(self.client.containers.get, container_ref)
        except NotFound:
            return None

    async def get_container_status(self, container_ref: str) -> str | None:
        """Return container status (running/exited/created) or None if missing."""
        container = await self.get_container(container_ref)
        if container is None:
            return None
        await asyncio.to_thread(container.reload)
        return container.status

    async def start_container(self, container_id: str) -> None:
        await self._ensure_client()
        container = self._get_container(container_id)
        await asyncio.to_thread(container.start)
        logger.debug(f"Started container {container_id[:12]}")

    async def stop_container(self, container_id: str, timeout: int = 10) -> None:
        await self._ensure_client()
        container = self._get_container(container_id)
        await asyncio.to_thread(container.stop, timeout=timeout)
        logger.debug(f"Stopped container {container_id[:12]}")

    async def remove_container(self, container_id: str, force: bool = True) -> None:
        await self._ensure_client()
        container = self._get_container(container_id)
        await asyncio.to_thread(container.remove, force=force)
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
        await self._ensure_client()
        container = self._get_container(container_id)
        timeout = timeout or self._config.timeout

        exec_config: dict[str, Any] = {
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

        exec_instance = await asyncio.to_thread(
            self.client.api.exec_create, container.id, **exec_config
        )

        try:
            output = await asyncio.wait_for(
                asyncio.to_thread(
                    self.client.api.exec_start, exec_instance["Id"], demux=True
                ),
                timeout=timeout,
            )
        except TimeoutError:
            logger.warning(
                "sandbox_command_timeout", extra={"operation.timeout": timeout}
            )
            return -1, "", f"Command timed out after {timeout} seconds"

        inspect_result = await asyncio.to_thread(
            self.client.api.exec_inspect, exec_instance["Id"]
        )
        exit_code = inspect_result.get("ExitCode", -1)
        stdout = output[0].decode("utf-8", errors="replace") if output[0] else ""
        stderr = output[1].decode("utf-8", errors="replace") if output[1] else ""

        return exit_code, stdout, stderr

    async def cleanup_all(self) -> None:
        for container_id in list(self._containers.keys()):
            try:
                await self.remove_container(container_id, force=True)
            except NotFound:
                self._containers.pop(container_id, None)

    def _get_container(self, container_id: str) -> Container:
        if container_id not in self._containers:
            try:
                self._containers[container_id] = self.client.containers.get(
                    container_id
                )
            except NotFound as e:
                raise KeyError(f"Container {container_id} not found") from e
        return self._containers[container_id]

    def __del__(self) -> None:
        if self._client:
            try:
                self._client.close()
            except Exception:
                # Log at debug level - __del__ may run during interpreter shutdown
                # when logging may not be available
                logger.debug("Failed to close Docker client", exc_info=True)
