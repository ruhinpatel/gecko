from __future__ import annotations

from pathlib import Path

import numpy as np
import qcelemental as qcel

from gecko.molecule.canonical import canonicalize_atom_order


_MADNESS_DIRECTIVES = {
    "eprec",
    "field",
    "no_orient",
    "psp_calc",
    "pure_ae",
    "symtol",
    "core_type",
    "units",
}


def _parse_madness_mol(path: Path) -> qcel.models.Molecule:
    lines = path.read_text(encoding="utf-8").splitlines()
    in_block = False
    units = "angstrom"
    symbols: list[str] = []
    coords: list[list[float]] = []

    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        lower = line.lower()
        if lower.startswith("geometry"):
            in_block = True
            continue
        if in_block and lower.startswith("end"):
            break
        if not in_block:
            continue

        tokens = line.split()
        if not tokens:
            continue

        key = tokens[0].lower()
        if key in _MADNESS_DIRECTIVES:
            if key == "units" and len(tokens) > 1:
                units = tokens[1].lower()
            continue

        if len(tokens) < 4:
            raise ValueError(f"Malformed coordinate line in {path}: {line!r}")

        symbol = tokens[0]
        try:
            x, y, z = (float(tokens[1]), float(tokens[2]), float(tokens[3]))
        except ValueError as exc:
            raise ValueError(f"Invalid coordinate line in {path}: {line!r}") from exc

        symbols.append(symbol)
        coords.append([x, y, z])

    if not symbols:
        raise ValueError(f"No geometry block found in {path}")

    geometry = np.asarray(coords, dtype=float)
    if units in {"atomic", "bohr"}:
        geometry = geometry * qcel.constants.bohr2angstroms

    symbols_sorted, geometry_sorted = canonicalize_atom_order(symbols, geometry, decimals=10)
    return qcel.models.Molecule(symbols=symbols_sorted, geometry=geometry_sorted)


def read_mol(path: Path) -> qcel.models.Molecule:
    """Read a MADNESS .mol file into a qcelemental Molecule."""
    mol_path = Path(path).expanduser().resolve()
    if not mol_path.exists():
        raise FileNotFoundError(f".mol file not found: {mol_path}")

    try:
        from quantumresponsepro import MADMolecule
    except Exception:
        MADMolecule = None

    if MADMolecule is not None:
        try:
            madmol = MADMolecule()
            madmol.from_molfile(mol_path)
            mol_json = madmol.to_json()
            symbols_sorted, geometry_sorted = canonicalize_atom_order(
                mol_json["symbols"], mol_json["geometry"], decimals=10
            )
            return qcel.models.Molecule(
                symbols=symbols_sorted,
                geometry=geometry_sorted,
            )
        except Exception as exc:
            raise ValueError(f"Failed to read .mol file with MADMolecule: {mol_path}") from exc

    try:
        return _parse_madness_mol(mol_path)
    except Exception as exc:
        raise ValueError(f"Failed to read .mol file: {mol_path}") from exc
