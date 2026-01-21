from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import qcelemental as qcel
from quantumresponsepro import MADMolecule


@dataclass(frozen=True)
class BetaPaperData:
    shg_pivot: pd.DataFrame
    molecules: Dict[str, qcel.models.Molecule]


def _repo_root_from_here() -> Path:
    # notebooks/scripts/data_loading.py -> notebooks/scripts -> notebooks -> repo
    return Path(__file__).resolve().parents[2]


@lru_cache(maxsize=1)
def load_shg_pivot(repo_root: Path | None = None) -> pd.DataFrame:
    """Load SHG beta tensor components and return a pivoted DataFrame.

    Returns a MultiIndex DataFrame indexed by (molecule, basis, omega)
    with 27 ijk columns.
    """
    if repo_root is None:
        repo_root = _repo_root_from_here()

    csv_path = repo_root / "data" / "csv_data" / "shg_ijk.csv"
    shg_ijk = pd.read_csv(csv_path)
    shg_pivot = shg_ijk.pivot_table(
        index=["molecule", "basis", "omega"], columns="ijk", values="Beta"
    )
    return shg_pivot


@lru_cache(maxsize=1)
def load_molecules(repo_root: Path | None = None) -> Dict[str, qcel.models.Molecule]:
    """Load .mol files under data/molecules and return a name->qcel Molecule map."""
    if repo_root is None:
        repo_root = _repo_root_from_here()

    mol_dir = repo_root / "data" / "molecules"
    molecules: Dict[str, qcel.models.Molecule] = {}

    for mol_file in sorted(mol_dir.glob("*.mol")):
        madmol = MADMolecule()
        madmol.from_molfile(mol_file)
        mol_json = madmol.to_json()

        symbols = mol_json["symbols"]
        geometry = mol_json["geometry"]
        molecules[mol_file.stem] = qcel.models.Molecule(symbols=symbols, geometry=geometry)

    return molecules


def load_all(repo_root: Path | None = None) -> BetaPaperData:
    """Load all data required for plotting."""
    return BetaPaperData(shg_pivot=load_shg_pivot(repo_root), molecules=load_molecules(repo_root))
