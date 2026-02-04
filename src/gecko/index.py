from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gecko
import pandas as pd
from gecko.core.model import Calculation
from gecko.enrich import enrich


@dataclass
class CalcFailure:
    path: str
    error: str


@dataclass
class CalcIndex:
    calcs: list[Calculation]
    failures: list[CalcFailure]

    def by_molecule(self, molecule_id: str) -> list[Calculation]:
        return [calc for calc in self.calcs if calc.meta.get("molecule_id") == molecule_id]

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for calc in self.calcs:
            props = sorted(
                k for k, v in calc.data.items()
                if v is not None and v != {} and v != [] and v != ()
            )
            rows.append(
                {
                    "code": calc.code,
                    "root": str(calc.root),
                    "basis": calc.meta.get("basis"),
                    "molecule_id": calc.meta.get("molecule_id"),
                    "properties_available": props,
                }
            )
        return pd.DataFrame(rows)

    @classmethod
    def from_dirs(
        cls,
        calc_dirs: Iterable[str | Path],
        strict: bool = False,
    ) -> "CalcIndex":
        calcs: list[Calculation] = []
        failures: list[CalcFailure] = []

        for d in calc_dirs:
            root = Path(d).expanduser().resolve()
            try:
                calc = gecko.load_calc(root)
                calc = enrich(calc)
                calcs.append(calc)
            except Exception as exc:
                failures.append(CalcFailure(path=str(root), error=f"{type(exc).__name__}: {exc}"))
                if strict:
                    raise

        return cls(calcs=calcs, failures=failures)
