from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import gecko
import pandas as pd
from gecko.core.model import Calculation
from gecko.enrich import enrich
from gecko.plugins.dalton.detect import detect_dalton, DaltonCandidate
from gecko.plugins.madness.detect import can_load as madness_can_load


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
        *,
        mol_root: str | Path | None = None,
        mol_map: str | Path | None = None,
        mol_file: str | Path | None = None,
        mol_dir: str | Path | None = None,
        strict: bool = False,
    ) -> "CalcIndex":
        calcs: list[Calculation] = []
        failures: list[CalcFailure] = []

        resolver = gecko.MoleculeResolver.from_sources(
            mol_file=mol_file,
            mol_dir=mol_dir or mol_root,
            mol_map=mol_map,
        )

        for d in calc_dirs:
            root = Path(d).expanduser().resolve()
            dalton_runs: list[DaltonCandidate] = detect_dalton(root)

            if dalton_runs:
                for run in dalton_runs:
                    try:
                        calc = gecko.load_calc(root, mol_resolver=resolver, run=run)
                        calc = enrich(
                            calc,
                            mol_root=mol_root,
                            mol_map=mol_map,
                            mol_file=mol_file,
                            mol_dir=mol_dir,
                        )
                        calcs.append(calc)
                    except Exception as exc:
                        run_id = run.get("meta", {}).get("out_file", "unknown")
                        failures.append(
                            CalcFailure(
                                path=f"{root}:{run_id}",
                                error=f"{type(exc).__name__}: {exc}",
                            )
                        )
                        if strict:
                            raise
                continue

            if madness_can_load(root):
                try:
                    calc = gecko.load_calc(root, mol_resolver=resolver)
                    calc = enrich(
                        calc,
                        mol_root=mol_root,
                        mol_map=mol_map,
                        mol_file=mol_file,
                        mol_dir=mol_dir,
                    )
                    calcs.append(calc)
                except Exception as exc:
                    failures.append(CalcFailure(path=str(root), error=f"{type(exc).__name__}: {exc}"))
                    if strict:
                        raise
                continue

            try:
                calc = gecko.load_calc(root, mol_resolver=resolver)
                calc = enrich(
                    calc,
                    mol_root=mol_root,
                    mol_map=mol_map,
                    mol_file=mol_file,
                    mol_dir=mol_dir,
                )
                calcs.append(calc)
            except Exception as exc:
                failures.append(CalcFailure(path=str(root), error=f"{type(exc).__name__}: {exc}"))
                if strict:
                    raise

        return cls(calcs=calcs, failures=failures)
