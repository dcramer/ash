"""Tests for the service management module."""

import os
import signal
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from ash.service.base import ServiceBackend, ServiceState, ServiceStatus
from ash.service.pid import (
    is_process_alive,
    read_pid_file,
    remove_pid_file,
    send_signal,
    write_pid_file,
)

# =============================================================================
# PID Utilities Tests
# =============================================================================


class TestPidUtilities:
    """Tests for PID file management."""

    def test_write_pid_file(self, tmp_path: Path):
        """Test writing PID file."""
        pid_path = tmp_path / "run" / "test.pid"
        write_pid_file(pid_path)

        assert pid_path.exists()
        content = pid_path.read_text().strip().split("\n")
        assert int(content[0]) == os.getpid()
        assert float(content[1]) > 0  # Start time

    def test_write_pid_file_custom_pid(self, tmp_path: Path):
        """Test writing PID file with custom PID."""
        pid_path = tmp_path / "run" / "test.pid"
        write_pid_file(pid_path, pid=12345)

        content = pid_path.read_text().strip().split("\n")
        assert content[0] == "12345"

    def test_read_pid_file_exists(self, tmp_path: Path):
        """Test reading existing PID file."""
        pid_path = tmp_path / "test.pid"
        current_pid = os.getpid()
        start_time = time.time()
        pid_path.write_text(f"{current_pid}\n{start_time}\n")

        proc_info = read_pid_file(pid_path)

        assert proc_info is not None
        assert proc_info.pid == current_pid
        assert proc_info.start_time == start_time
        assert proc_info.alive is True  # Current process is alive

    def test_read_pid_file_not_exists(self, tmp_path: Path):
        """Test reading non-existent PID file."""
        pid_path = tmp_path / "nonexistent.pid"
        proc_info = read_pid_file(pid_path)
        assert proc_info is None

    def test_read_pid_file_dead_process(self, tmp_path: Path):
        """Test reading PID file for dead process."""
        pid_path = tmp_path / "test.pid"
        # Use a PID that's unlikely to be running
        pid_path.write_text("999999\n0\n")

        proc_info = read_pid_file(pid_path)

        assert proc_info is not None
        assert proc_info.pid == 999999
        assert proc_info.alive is False

    def test_remove_pid_file(self, tmp_path: Path):
        """Test removing PID file."""
        pid_path = tmp_path / "test.pid"
        pid_path.write_text("12345\n0\n")

        assert pid_path.exists()
        remove_pid_file(pid_path)
        assert not pid_path.exists()

    def test_remove_pid_file_not_exists(self, tmp_path: Path):
        """Test removing non-existent PID file (no error)."""
        pid_path = tmp_path / "nonexistent.pid"
        remove_pid_file(pid_path)  # Should not raise

    def test_is_process_alive_current(self):
        """Test checking if current process is alive."""
        assert is_process_alive(os.getpid()) is True

    def test_is_process_alive_dead(self):
        """Test checking if dead process is alive."""
        assert is_process_alive(999999) is False

    def test_send_signal_success(self):
        """Test sending signal to current process."""
        # SIGCONT is harmless and can be sent to self
        result = send_signal(os.getpid(), signal.SIGCONT)
        assert result is True

    def test_send_signal_failure(self):
        """Test sending signal to non-existent process."""
        result = send_signal(999999, signal.SIGTERM)
        assert result is False


# =============================================================================
# Backend Detection Tests
# =============================================================================


class TestBackendDetection:
    """Tests for backend detection."""

    def test_detect_backend_returns_backend(self):
        """Test that detect_backend returns a valid backend."""
        from ash.service.backends import detect_backend

        backend = detect_backend()
        assert isinstance(backend, ServiceBackend)
        assert backend.name in ("systemd", "launchd", "generic")

    def test_get_backend_generic(self):
        """Test getting generic backend by name."""
        from ash.service.backends import get_backend

        backend = get_backend("generic")
        assert backend.name == "generic"
        assert backend.is_available is True

    def test_get_backend_invalid(self):
        """Test getting invalid backend raises error."""
        from ash.service.backends import get_backend

        with pytest.raises(ValueError, match="Unknown backend"):
            get_backend("invalid")

    def test_get_backend_auto_detect(self):
        """Test auto-detecting backend."""
        from ash.service.backends import get_backend

        backend = get_backend(None)
        assert isinstance(backend, ServiceBackend)


# =============================================================================
# Generic Backend Tests
# =============================================================================


class TestGenericBackend:
    """Tests for the generic fallback backend."""

    @pytest.fixture
    def backend(self, tmp_path: Path, monkeypatch):
        """Create a generic backend with temporary paths."""
        from ash.service.backends.generic import GenericBackend

        # Override paths to use tmp_path
        monkeypatch.setattr(
            "ash.service.backends.generic.get_pid_path",
            lambda: tmp_path / "run" / "ash.pid",
        )
        monkeypatch.setattr(
            "ash.service.backends.generic.get_service_log_path",
            lambda: tmp_path / "logs" / "service.log",
        )

        return GenericBackend()

    def test_name(self, backend):
        """Test backend name."""
        assert backend.name == "generic"

    def test_is_available(self, backend):
        """Test generic backend is always available."""
        assert backend.is_available is True

    def test_supports_install(self, backend):
        """Test generic backend doesn't support install."""
        assert backend.supports_install is False

    @pytest.mark.asyncio
    async def test_status_stopped(self, backend, tmp_path: Path):
        """Test status when service is stopped."""
        status = await backend.status()
        assert status.state == ServiceState.STOPPED

    @pytest.mark.asyncio
    async def test_status_running(self, backend, tmp_path: Path):
        """Test status when service is running."""
        # Create a PID file for the current process
        pid_path = tmp_path / "run" / "ash.pid"
        write_pid_file(pid_path)

        status = await backend.status()
        assert status.state == ServiceState.RUNNING
        assert status.pid == os.getpid()

    @pytest.mark.asyncio
    async def test_stop_not_running(self, backend):
        """Test stop when service is not running."""
        result = await backend.stop()
        assert result is True

    @pytest.mark.asyncio
    async def test_install_raises(self, backend):
        """Test install raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            await backend.install()

    @pytest.mark.asyncio
    async def test_uninstall_succeeds(self, backend):
        """Test uninstall succeeds (no-op)."""
        result = await backend.uninstall()
        assert result is True

    def test_get_log_source(self, backend, tmp_path: Path):
        """Test get_log_source returns path."""
        log_source = backend.get_log_source()
        assert isinstance(log_source, Path)
        assert "service.log" in str(log_source)


# =============================================================================
# ServiceManager Tests
# =============================================================================


class TestServiceManager:
    """Tests for the ServiceManager class."""

    @pytest.fixture
    def mock_backend(self):
        """Create a mock backend."""
        backend = MagicMock(spec=ServiceBackend)
        backend.name = "mock"
        backend.supports_install = True
        return backend

    @pytest.fixture
    def manager(self, mock_backend):
        """Create a manager with mock backend."""
        from ash.service.manager import ServiceManager

        return ServiceManager(backend=mock_backend)

    @pytest.mark.asyncio
    async def test_start_success(self, manager, mock_backend):
        """Test starting service successfully."""
        mock_backend.status = AsyncMock(
            side_effect=[
                ServiceStatus(state=ServiceState.STOPPED),
                ServiceStatus(state=ServiceState.RUNNING, pid=12345),
            ]
        )
        mock_backend.start = AsyncMock(return_value=True)

        success, message = await manager.start()

        assert success is True
        assert "started" in message.lower()
        mock_backend.start.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_already_running(self, manager, mock_backend):
        """Test starting when already running."""
        mock_backend.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.RUNNING, pid=12345)
        )

        success, message = await manager.start()

        assert success is False
        assert "already running" in message.lower()

    @pytest.mark.asyncio
    async def test_stop_success(self, manager, mock_backend):
        """Test stopping service successfully."""
        mock_backend.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.RUNNING, pid=12345)
        )
        mock_backend.stop = AsyncMock(return_value=True)

        success, message = await manager.stop()

        assert success is True
        assert "stopped" in message.lower()
        mock_backend.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self, manager, mock_backend):
        """Test stopping when already stopped."""
        mock_backend.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.STOPPED)
        )

        success, message = await manager.stop()

        assert success is True
        assert "already stopped" in message.lower()

    @pytest.mark.asyncio
    async def test_restart_success(self, manager, mock_backend):
        """Test restarting service successfully."""
        mock_backend.restart = AsyncMock(return_value=True)
        mock_backend.status = AsyncMock(
            return_value=ServiceStatus(state=ServiceState.RUNNING, pid=12345)
        )

        success, message = await manager.restart()

        assert success is True
        assert "restarted" in message.lower()

    @pytest.mark.asyncio
    async def test_status(self, manager, mock_backend):
        """Test getting service status."""
        expected_status = ServiceStatus(
            state=ServiceState.RUNNING,
            pid=12345,
            uptime_seconds=3600.0,
        )
        mock_backend.status = AsyncMock(return_value=expected_status)

        status = await manager.status()

        assert status == expected_status

    @pytest.mark.asyncio
    async def test_install_success(self, manager, mock_backend):
        """Test installing service successfully."""
        mock_backend.install = AsyncMock(return_value=True)

        success, message = await manager.install()

        assert success is True
        assert "installed" in message.lower()

    @pytest.mark.asyncio
    async def test_install_not_supported(self, manager, mock_backend):
        """Test installing when not supported."""
        mock_backend.supports_install = False

        success, message = await manager.install()

        assert success is False
        assert "not supported" in message.lower()

    @pytest.mark.asyncio
    async def test_uninstall_success(self, manager, mock_backend):
        """Test uninstalling service successfully."""
        mock_backend.uninstall = AsyncMock(return_value=True)

        success, message = await manager.uninstall()

        assert success is True
        assert "uninstalled" in message.lower()


# =============================================================================
# CLI Integration Tests
# =============================================================================


class TestServiceCLI:
    """Tests for service CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create a Typer CLI test runner."""
        from typer.testing import CliRunner

        return CliRunner()

    def test_service_status_command(self, cli_runner, monkeypatch):
        """Test 'ash service status' command."""
        from ash.cli.app import app

        # Mock the ServiceManager
        mock_status = ServiceStatus(state=ServiceState.STOPPED)

        async def mock_status_fn():
            return mock_status

        mock_manager = MagicMock()
        mock_manager.status = mock_status_fn
        mock_manager.backend_name = "generic"

        monkeypatch.setattr(
            "ash.service.ServiceManager",
            lambda backend=None: mock_manager,
        )

        result = cli_runner.invoke(app, ["service", "status"])

        assert result.exit_code == 0
        assert "stopped" in result.output.lower()

    def test_service_help(self, cli_runner):
        """Test 'ash service --help' command."""
        from ash.cli.app import app

        result = cli_runner.invoke(app, ["service", "--help"])

        assert result.exit_code == 0
        assert "start" in result.output
        assert "stop" in result.output
        assert "status" in result.output
        assert "logs" in result.output
        assert "install" in result.output
        assert "uninstall" in result.output


# =============================================================================
# Runtime State Tests
# =============================================================================


class TestRuntimeState:
    """Tests for runtime state management."""

    def test_runtime_state_to_json(self):
        """Test serializing RuntimeState to JSON."""
        from ash.service.runtime import RuntimeState

        state = RuntimeState(
            started_at="2024-01-01T00:00:00+00:00",
            model="claude-sonnet-4-20250514",
            sandbox_image="ash-sandbox:latest",
            sandbox_network="bridge",
            sandbox_runtime="runc",
            workspace_path="/path/to/workspace",
            workspace_access="rw",
            source_access="none",
            sessions_access="ro",
            chats_access="ro",
        )

        json_str = state.to_json()
        assert "claude-sonnet" in json_str
        assert "ash-sandbox:latest" in json_str
        assert "bridge" in json_str

    def test_runtime_state_from_json(self):
        """Test deserializing RuntimeState from JSON."""
        from ash.service.runtime import RuntimeState

        json_str = """{
            "started_at": "2024-01-01T00:00:00+00:00",
            "model": "claude-sonnet-4-20250514",
            "sandbox_image": "ash-sandbox:latest",
            "sandbox_network": "bridge",
            "sandbox_runtime": "runc",
            "workspace_path": "/path/to/workspace",
            "workspace_access": "rw",
            "source_access": "none",
            "sessions_access": "ro",
            "chats_access": "ro"
        }"""

        state = RuntimeState.from_json(json_str)
        assert state.model == "claude-sonnet-4-20250514"
        assert state.sandbox_image == "ash-sandbox:latest"
        assert state.sandbox_network == "bridge"
        assert state.workspace_access == "rw"

    def test_write_and_read_runtime_state(self, tmp_path: Path, monkeypatch):
        """Test writing and reading runtime state from disk."""
        from ash.service.runtime import (
            RuntimeState,
            read_runtime_state,
            write_runtime_state,
        )

        # Patch get_run_path to use tmp_path
        monkeypatch.setattr(
            "ash.service.runtime.get_run_path",
            lambda: tmp_path / "run",
        )

        state = RuntimeState(
            started_at="2024-01-01T00:00:00+00:00",
            model="claude-sonnet-4-20250514",
            sandbox_image="ash-sandbox:latest",
            sandbox_network="bridge",
            sandbox_runtime="runc",
            workspace_path="/path/to/workspace",
            workspace_access="rw",
            source_access="none",
            sessions_access="ro",
            chats_access="ro",
        )

        write_runtime_state(state)
        read_state = read_runtime_state()

        assert read_state is not None
        assert read_state.model == state.model
        assert read_state.sandbox_image == state.sandbox_image

    def test_read_runtime_state_not_exists(self, tmp_path: Path, monkeypatch):
        """Test reading runtime state when file doesn't exist."""
        from ash.service.runtime import read_runtime_state

        monkeypatch.setattr(
            "ash.service.runtime.get_run_path",
            lambda: tmp_path / "nonexistent",
        )

        state = read_runtime_state()
        assert state is None

    def test_remove_runtime_state(self, tmp_path: Path, monkeypatch):
        """Test removing runtime state file."""
        from ash.service.runtime import (
            RuntimeState,
            read_runtime_state,
            remove_runtime_state,
            write_runtime_state,
        )

        monkeypatch.setattr(
            "ash.service.runtime.get_run_path",
            lambda: tmp_path / "run",
        )

        state = RuntimeState(
            started_at="2024-01-01T00:00:00+00:00",
            model="test",
            sandbox_image="test",
            sandbox_network="bridge",
            sandbox_runtime="runc",
            workspace_path="/path",
            workspace_access="rw",
            source_access="none",
            sessions_access="ro",
            chats_access="ro",
        )

        write_runtime_state(state)
        assert read_runtime_state() is not None

        remove_runtime_state()
        assert read_runtime_state() is None

    def test_remove_runtime_state_not_exists(self, tmp_path: Path, monkeypatch):
        """Test removing runtime state when file doesn't exist."""
        from ash.service.runtime import remove_runtime_state

        monkeypatch.setattr(
            "ash.service.runtime.get_run_path",
            lambda: tmp_path / "nonexistent",
        )

        # Should not raise
        remove_runtime_state()
