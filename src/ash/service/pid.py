"""PID file management utilities."""

import os
import signal
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ProcessInfo:
    """Process information from PID file."""

    pid: int
    start_time: float
    alive: bool


def write_pid_file(pid_path: Path, pid: int | None = None) -> None:
    """Write current process PID to file.

    Args:
        pid_path: Path to the PID file.
        pid: Process ID to write. Defaults to current process.
    """
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    pid_path.write_text(f"{pid or os.getpid()}\n{time.time()}\n")


def read_pid_file(pid_path: Path) -> ProcessInfo | None:
    """Read PID file and check if process is alive.

    Args:
        pid_path: Path to the PID file.

    Returns:
        ProcessInfo if file exists, None otherwise.
    """
    if not pid_path.exists():
        return None

    try:
        content = pid_path.read_text().strip().split("\n")
        pid = int(content[0])
        start_time = float(content[1]) if len(content) > 1 else 0.0
        alive = is_process_alive(pid)
        return ProcessInfo(pid=pid, start_time=start_time, alive=alive)
    except (ValueError, IndexError):
        return None


def remove_pid_file(pid_path: Path) -> None:
    """Remove PID file if it exists.

    Args:
        pid_path: Path to the PID file.
    """
    pid_path.unlink(missing_ok=True)


def is_process_alive(pid: int) -> bool:
    """Check if a process with given PID is alive.

    Args:
        pid: Process ID to check.

    Returns:
        True if process exists and is running.
    """
    try:
        os.kill(pid, 0)  # Signal 0 checks existence without sending signal
        return True
    except OSError:
        return False


def send_signal(pid: int, sig: signal.Signals) -> bool:
    """Send signal to process.

    Args:
        pid: Process ID to signal.
        sig: Signal to send.

    Returns:
        True if signal was sent successfully.
    """
    try:
        os.kill(pid, sig)
        return True
    except OSError:
        return False


def get_process_info(pid: int) -> dict[str, float] | None:
    """Get process resource information.

    Args:
        pid: Process ID to query.

    Returns:
        Dict with memory_mb and cpu_percent, or None if unavailable.
    """
    try:
        import psutil

        proc = psutil.Process(pid)
        mem_info = proc.memory_info()
        return {
            "memory_mb": mem_info.rss / (1024 * 1024),
            "cpu_percent": proc.cpu_percent(interval=0.1),
        }
    except Exception:
        # psutil not installed or process not found
        return None
