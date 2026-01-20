from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import qcelemental as qcel


@dataclass
class Calculation:
    """
    Minimal container for a parsed calculation.

    Migration-first design:
    - 'data' can hold raw legacy parser outputs (dicts, etc.)
    - 'artifacts' lists discovered files
    - 'meta' is freeform convenience metadata
    """
    code: str                  # "madness" or "dalton"
    root: Path
    artifacts: dict[str, Path] = field(default_factory=dict)
    data: dict[str, Any] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)
    molecule: Optional[qcel.models.Molecule] = None

    def __repr__(self) -> str:
        keys = ", ".join(sorted(self.data.keys())) if self.data else "-"
        return f"Calculation(code={self.code!r}, root={str(self.root)!r}, data_keys={keys})"
