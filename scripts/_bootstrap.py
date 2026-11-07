from __future__ import annotations

import sys
from pathlib import Path


def add_project_src_to_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    src = project_root / "src"
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
