from __future__ import annotations

import hashlib

import numpy as np
import qcelemental as qcel


def _geometry_angstrom(mol: qcel.models.Molecule) -> np.ndarray:
    geometry = np.asarray(mol.geometry, dtype=np.float64, order="C").reshape(-1, 3)
    units = getattr(mol, "units", None)
    if isinstance(units, str) and units.lower() in ("bohr", "atomic", "au"):
        geometry = geometry * qcel.constants.bohr2angstroms
    return np.asarray(geometry, dtype=np.float64, order="C")


def compute_molecule_id(mol: qcelemental.models.Molecule) -> str:
    symbols_bytes = b"\0".join(str(sym).encode("utf-8") for sym in mol.symbols)
    geometry_bytes = _geometry_angstrom(mol).tobytes()

    charge = np.int64(int(getattr(mol, "charge", 0))).tobytes()
    multiplicity = np.int64(int(getattr(mol, "multiplicity", 1))).tobytes()

    hasher = hashlib.sha256()
    hasher.update(symbols_bytes)
    hasher.update(geometry_bytes)
    hasher.update(charge)
    hasher.update(multiplicity)
    return f"mol:{hasher.hexdigest()[:16]}"
