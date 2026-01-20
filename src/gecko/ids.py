from __future__ import annotations

from collections import Counter
from hashlib import sha1
from typing import Optional

import numpy as np
import qcelemental as qcel

from gecko.core.model import Calculation


def _geometry_angstrom(mol: qcel.models.Molecule) -> np.ndarray:
    geometry = np.asarray(mol.geometry, dtype=float).reshape(-1, 3)
    units = getattr(mol, "units", None)
    if units in ("bohr", "atomic"):
        geometry = geometry * qcel.constants.bohr2angstroms
    return geometry


def geom_id(molecule: qcel.models.Molecule) -> str:
    symbols = list(molecule.symbols)
    geometry = _geometry_angstrom(molecule)
    geometry = np.round(geometry, 6)

    payload = {
        "symbols": symbols,
        "geometry": geometry.tolist(),
    }
    digest = sha1(str(payload).encode("utf-8")).hexdigest()[:16]
    return f"geom_{digest}"


def calc_id(calc: Calculation) -> str:
    root = calc.root.resolve()
    base = f"{calc.code}|{root}"
    digest = sha1(base.encode("utf-8")).hexdigest()[:16]
    return f"calc_{digest}"


def mol_id_from_symbols(symbols: list[str]) -> str:
    counts = Counter(str(s) for s in symbols)
    formula = "".join(f"{sym}{counts[sym] if counts[sym] > 1 else ''}" for sym in sorted(counts))
    return formula


def mol_id(calc: Calculation) -> Optional[str]:
    label = calc.meta.get("molecule_name") or calc.meta.get("molecule")
    if label:
        return str(label)
    if calc.molecule is not None:
        return mol_id_from_symbols(list(calc.molecule.symbols))
    return None
