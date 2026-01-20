from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import pandas as pd

from gecko.core.model import Calculation
from gecko.ids import geom_id, mol_id
from gecko.tables.extractors import (
    extract_alpha,
    extract_beta,
    extract_dipole,
    extract_energy,
)


@dataclass
class TableBuilder:
    calcs: list[Calculation]

    def build_geometries(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for calc in self.calcs:
            qmol = calc.molecule
            if qmol is None:
                continue

            gid = calc.meta.get("geom_id") or geom_id(qmol)
            if gid in seen:
                continue
            seen.add(gid)

            geometry = np.asarray(qmol.geometry, dtype=float).reshape(-1, 3)
            rows.append(
                {
                    "geom_id": gid,
                    "mol_id": calc.meta.get("mol_id") or mol_id(calc),
                    "label": calc.meta.get("label"),
                    "symbols": list(qmol.symbols),
                    "geometry_angstrom": geometry.tolist(),
                    "charge": getattr(qmol, "charge", None),
                    "multiplicity": getattr(qmol, "multiplicity", None),
                    "source_calc_id": calc.meta.get("calc_id"),
                    "source_root": str(calc.root),
                }
            )

        return pd.DataFrame(rows)

    def build_beta(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_beta(calc))
        return pd.DataFrame(rows)

    def build_alpha(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_alpha(calc))
        return pd.DataFrame(rows)

    def build_energy(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_energy(calc))
        return pd.DataFrame(rows)

    def build_dipole(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_dipole(calc))
        return pd.DataFrame(rows)
