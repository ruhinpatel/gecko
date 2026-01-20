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
    out_files = sorted(calc_dir.glob("*.out"))
    dalton_upper = calc_dir / "DALTON.OUT"
    if dalton_upper.exists() and dalton_upper not in out_files:
        out_files.insert(0, dalton_upper)
    quad_files = [
        p for p in out_files if any(tok in p.name.lower() for tok in ("quad", "qr", "response"))
    ]

    dalton_out = next((p for p in out_files if p.name == "DALTON.OUT"), None)
    if dalton_out is not None:
        artifacts = {"out": dalton_out, "output": dalton_out}
        if quad_files:
            artifacts["dalton_quad_out"] = quad_files[0]
        runs.append(
            {
                "code": "dalton",
                "root": calc_dir,
                "artifacts": artifacts,
                "meta": {"out_file": dalton_out.name, "stem": dalton_out.stem},
            }
        )

    for out_path in out_files:
        if out_path == dalton_out:
            continue
        if out_path in quad_files:
            continue
        runs.append(
            {
                "code": "dalton",
                "root": calc_dir,
                "artifacts": {"out": out_path, "output": out_path},
                "meta": {"out_file": out_path.name, "stem": out_path.stem},
            }
        )

    if not runs and quad_files:
        for quad_path in quad_files:
            runs.append(
                {
                    "code": "dalton",
                    "root": calc_dir,
                    "artifacts": {"out": quad_path, "output": quad_path, "dalton_quad_out": quad_path},
                    "meta": {"out_file": quad_path.name, "stem": quad_path.stem},
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
