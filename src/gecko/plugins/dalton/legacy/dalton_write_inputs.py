from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import qcelemental as qcel
import numpy as np


def _has_no_orient(extras: Optional[dict], default: bool) -> bool:
    if not isinstance(extras, dict):
        return default

    if extras.get("no_orient") is not None:
        value = extras["no_orient"]
        return str(value).lower() == "true" if isinstance(value, str) else bool(value)

    for key in ("parameters", "madness_parameters"):
        params = extras.get(key)
        if isinstance(params, dict) and params.get("no_orient") is not None:
            value = params["no_orient"]
            return (
                str(value).lower() == "true" if isinstance(value, str) else bool(value)
            )

    return default


def to_string(
    molecule: qcel.models.Molecule,
    basis: str,
    *,
    charge: Optional[int] = None,
    units: str = "Bohr",
    atom_format: str = None,
    ghost_format: str = None,
    width: int = 17,
    prec: int = 12,
) -> str:
    r"""Formamat a string representation of QM moleucule.  Based on qcelemental.molparse.to_string
    To match qcelemental implementation we need to additionally deal with units, width, precision, etc.


    """
    if not isinstance(molecule, qcel.models.Molecule):
        raise TypeError("molecule must be an instance of qcelemental.models.Molecule")

    dalton_lines = ["BASIS", str(basis), "blah", "blah"]

    extras = getattr(molecule, "extras", None) or {}
    # default units in qcel is Bohr, if user provides angstrom we need to convert the coordinates
    factor = 1.0
    if units.lower() in ["angstrom"]:
        units = "Angstrom"
        factor = qcel.constants.conversion_factor("bohr", "angstrom")
    else:
        units = " "
    
    geom = np.asarray(molecule.geometry).reshape((-1, 3)) * factor
    charge_value = molecule.molecular_charge if charge is None else charge
    charge_value = int(round(charge_value))

    atom_coords: dict[str, list[str]] = {}
    coordinates = geom.reshape((-1, 3))
    fxyz = """{:>{width}.{prec}f}"""
    for symbol, coord in zip(molecule.symbols, coordinates):
        formatted = " ".join(fxyz.format(float(c), width=width, prec=prec) for c in coord)
        atom_coords.setdefault(symbol, []).append(formatted)

    general_line = f"Atomtype={len(atom_coords)} {units} Charge={charge_value}"
    if _has_no_orient(extras, bool(getattr(molecule, "fix_orientation", False))):
        general_line += " Nosymmetry"
    dalton_lines.append(general_line)

    # from qcelemental.molparse.to_string
    for atom, coords in atom_coords.items():
        atomic_number = qcel.periodictable.to_atomic_number(atom)
        dalton_lines.append(f"Charge={atomic_number} Atoms={len(coords)}")
        suffix = ord("a")
        for coord in coords:
            dalton_lines.append(
                f"{atom}_{chr(suffix)} {coord}"
            )
            suffix += 1

    return "\n".join(dalton_lines)
