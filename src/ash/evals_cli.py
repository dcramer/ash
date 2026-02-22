from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    """Ensure local `evals/` package is importable from the repository checkout."""
    repo_root = Path(__file__).resolve().parents[2]
    if not (repo_root / "evals").exists():
        return
    root_str = str(repo_root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)


def main() -> None:
    _ensure_repo_root_on_path()
    from evals.cli import app

    app()
