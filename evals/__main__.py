from __future__ import annotations

import sys
from pathlib import Path

# When executed as a script path (e.g., `uv run evals ...`), Python can place the
# evals directory itself on sys.path, which shadows stdlib modules like `types`.
# Normalize sys.path to include the project root instead.
this_dir = Path(__file__).resolve().parent
project_root = this_dir.parent
sys.path = [p for p in sys.path if Path(p).resolve() != this_dir]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from evals.cli import app

app()
