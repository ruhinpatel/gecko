"""Geometry acquisition from PubChem or local files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import qcelemental as qcel

PUBCHEM_BASE = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"

# Atomic number → symbol for common elements (Z 1-54)
_Z_TO_SYMBOL: dict[int, str] = {
    1: "H", 2: "He", 3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O",
    9: "F", 10: "Ne", 11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P",
    16: "S", 17: "Cl", 18: "Ar", 19: "K", 20: "Ca", 21: "Sc", 22: "Ti",
    23: "V", 24: "Cr", 25: "Mn", 26: "Fe", 27: "Co", 28: "Ni", 29: "Cu",
    30: "Zn", 31: "Ga", 32: "Ge", 33: "As", 34: "Se", 35: "Br", 36: "Kr",
    37: "Rb", 38: "Sr", 39: "Y", 40: "Zr", 41: "Nb", 42: "Mo", 43: "Tc",
    44: "Ru", 45: "Rh", 46: "Pd", 47: "Ag", 48: "Cd", 49: "In", 50: "Sn",
    51: "Sb", 52: "Te", 53: "I", 54: "Xe",
}


def fetch_geometry(name: str, *, source: str = "pubchem") -> qcel.models.Molecule:
    """Fetch molecular geometry by name.

    Parameters
    ----------
    name : str
        Molecule name or formula (e.g. "SO2", "water", "sulfur dioxide").
    source : str
        Geometry source: "pubchem" (default).

    Returns
    -------
    qcelemental.models.Molecule
        Molecule with geometry in Bohr (qcelemental convention).
    """
    if source == "pubchem":
        return _fetch_pubchem(name)
    raise ValueError(f"Unknown source {source!r}. Currently only 'pubchem' is supported.")


def load_geometry_from_file(path: Path) -> qcel.models.Molecule:
    """Load geometry from a local file.

    Supports:
    - ``.xyz`` — standard XYZ format (Angstrom)
    - ``.mol`` — MADNESS geometry format (atomic units / Bohr)
    """
    path = Path(path)
    if path.suffix == ".xyz":
        return _load_xyz(path)
    if path.suffix == ".mol":
        return _load_madness_mol(path)
    raise ValueError(
        f"Unsupported file extension {path.suffix!r}. Use .xyz or .mol"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_pubchem(name: str) -> qcel.models.Molecule:
    """Fetch geometry from PubChem by molecule name, trying 3D then 2D."""
    try:
        import requests
    except ImportError:
        raise ImportError(
            "The 'requests' package is required for PubChem fetch.\n"
            "Install it with:  pip install requests"
        )

    # Resolve name → CID
    url = f"{PUBCHEM_BASE}/compound/name/{name}/cids/JSON"
    r = requests.get(url, timeout=15)
    if r.status_code == 404:
        raise ValueError(f"Molecule {name!r} not found in PubChem.")
    r.raise_for_status()
    cid = r.json()["IdentifierList"]["CID"][0]

    # Fetch 3D conformer; fall back to 2D if unavailable
    for record_type in ("3d", "2d"):
        url = f"{PUBCHEM_BASE}/compound/cid/{cid}/JSON?record_type={record_type}"
        r = requests.get(url, timeout=15)
        if r.status_code == 404:
            continue
        r.raise_for_status()
        data = r.json()
        compound = data["PC_Compounds"][0]

        elements: list[int] = compound["atoms"]["element"]
        conformer = compound["coords"][0]["conformers"][0]
        xs: list[float] = conformer["x"]
        ys: list[float] = conformer["y"]
        zs: list[float] = conformer.get("z", [0.0] * len(xs))

        symbols = [_atomic_number_to_symbol(z) for z in elements]
        geom_ang = np.array(list(zip(xs, ys, zs)), dtype=float)
        geom_bohr = geom_ang * qcel.constants.conversion_factor("angstrom", "bohr")

        return qcel.models.Molecule(
            symbols=symbols,
            geometry=geom_bohr.flatten().tolist(),
            name=name,
            molecular_charge=0,
            molecular_multiplicity=1,
        )

    raise RuntimeError(f"Could not retrieve a conformer for CID {cid} ({name!r}).")


def _load_xyz(path: Path) -> qcel.models.Molecule:
    """Load a standard XYZ file (coordinates in Angstrom)."""
    lines = path.read_text().splitlines()
    n_atoms = int(lines[0].strip())
    symbols = []
    coords_ang = []
    for line in lines[2 : 2 + n_atoms]:
        parts = line.split()
        symbols.append(parts[0])
        coords_ang.append([float(x) for x in parts[1:4]])
    geom_bohr = (
        np.array(coords_ang) * qcel.constants.conversion_factor("angstrom", "bohr")
    )
    return qcel.models.Molecule(
        symbols=symbols,
        geometry=geom_bohr.flatten().tolist(),
        name=path.stem,
    )


def _load_madness_mol(path: Path) -> qcel.models.Molecule:
    """Load a MADNESS .mol file."""
    from gecko.plugins.madness.legacy.madness_molecule import MADMolecule

    mad_mol = MADMolecule()
    mad_mol.from_molfile(path)

    geom = np.array(mad_mol.geometry, dtype=float)
    units = mad_mol.parameters.units.lower()
    if units in ("angstrom",):
        geom = geom * qcel.constants.conversion_factor("angstrom", "bohr")

    return qcel.models.Molecule(
        symbols=mad_mol.symbols,
        geometry=geom.flatten().tolist(),
        name=path.stem,
    )


def _atomic_number_to_symbol(z: int) -> str:
    symbol = _Z_TO_SYMBOL.get(z)
    if symbol is None:
        raise ValueError(f"Atomic number {z} not in lookup table (only Z=1–54 supported).")
    return symbol
