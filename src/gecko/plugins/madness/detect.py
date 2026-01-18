from __future__ import annotations

from pathlib import Path


def can_load(path: Path) -> bool:
    """
    Detect MADNESS run directories.

    Supports:
      - MADQC style: *.calc_info.json
      - Legacy molresponse style: output.json
      - Optional: responses/metadata.json
    """
    if not path.exists() or not path.is_dir():
        return False

    # MADQC marker
    if any(path.glob("*.calc_info.json")):
        return True

    # Legacy molresponse marker
    if (path / "output.json").exists():
        return True

    # Optional common marker
    if (path / "responses" / "metadata.json").exists():
        return True

    return False
