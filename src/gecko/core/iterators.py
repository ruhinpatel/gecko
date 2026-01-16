from __future__ import annotations

from pathlib import Path
from typing import Iterator

from gecko.plugins.madness.detect import can_load as madness_can_load
from gecko.plugins.dalton.detect import can_load as dalton_can_load


def iter_calc_dirs(root: str | Path) -> Iterator[Path]:
    """
    Walk a directory tree and yield subdirectories that look like MADNESS or DALTON runs.
    Read-only, layout-agnostic.
    """
    root = Path(root).expanduser().resolve()
    if not root.exists():
        return

    for p in root.rglob("*"):
        if not p.is_dir():
            continue
        if madness_can_load(p) or dalton_can_load(p):
            yield p
