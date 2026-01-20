from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class DaltonCandidate(TypedDict):
    code: str
    root: Path
    artifacts: dict[str, Path]
    meta: dict[str, str]


def detect_dalton(calc_dir: Path) -> list[DaltonCandidate]:
    if not calc_dir.exists() or not calc_dir.is_dir():
        return []

    runs: list[DaltonCandidate] = []
    for out_path in sorted(calc_dir.glob("*.out")):
        runs.append(
            {
                "code": "dalton",
                "root": calc_dir,
                "artifacts": {"out": out_path, "output": out_path},
                "meta": {"out_file": out_path.name, "stem": out_path.stem},
            }
        )
    return runs


def can_load(path: Path) -> bool:
    """
    Heuristic detection for DALTON outputs (migration fixtures first).
    """
    if not path.exists() or not path.is_dir():
        return False
    return bool(detect_dalton(path))
