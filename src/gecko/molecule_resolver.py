from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Optional, Protocol

import numpy as np
import qcelemental as qcel

from gecko.core.model import Calculation
from gecko.molecule.canonical import canonicalize_atom_order
from gecko.mol.io import read_mol
from gecko.mol.resolver import MoleculeResolver, mol_label_from_calc


class MoleculeResolverProtocol(Protocol):
    def resolve(self, *, molecule_key: str | None, calc: Calculation) -> Optional[qcel.models.Molecule]:
        ...


class FromProvidedMolDir:
    def __init__(self, mol_dir: Path | str):
        self.mol_dir = Path(mol_dir).expanduser().resolve()

    def resolve(self, *, molecule_key: str | None, calc: Calculation) -> Optional[qcel.models.Molecule]:
        if molecule_key is None:
            return None
        candidate = self.mol_dir / f"{molecule_key}.mol"
        if candidate.exists():
            calc.meta.setdefault("molecule_source", "resolver:mol_dir")
            calc.meta.setdefault("molecule_path", str(candidate))
            return read_mol(candidate)
        return None


class FromMappingFile:
    def __init__(self, mapping_json: Path | str):
        self.mapping_json = Path(mapping_json).expanduser().resolve()
        self._mapping = _load_map(self.mapping_json)

    def resolve(self, *, molecule_key: str | None, calc: Calculation) -> Optional[qcel.models.Molecule]:
        keys = [str(calc.root), calc.meta.get("run_id"), molecule_key]
        for key in keys:
            if key is None:
                continue
            if str(key) in self._mapping:
                mol_path = Path(self._mapping[str(key)]).expanduser().resolve()
                calc.meta.setdefault("molecule_source", "resolver:map")
                calc.meta.setdefault("molecule_path", str(mol_path))
                return read_mol(mol_path)
        return None


@lru_cache(maxsize=8)
def _load_map(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"mol_map JSON must be an object: {path}")
    return {str(k): str(v) for k, v in data.items()}


def _molecule_from_calc_info(calc: Calculation) -> Optional[qcel.models.Molecule]:
    info = calc.data.get("calc_info")
    if not isinstance(info, dict):
        return None

    mol_block = info.get("molecule")
    if not isinstance(mol_block, dict):
        return None

    symbols = mol_block.get("symbols")
    geometry = mol_block.get("geometry")
    if symbols is None or geometry is None:
        return None

    units = mol_block.get("units") or mol_block.get("parameters", {}).get("units")
    coords = np.asarray(geometry, dtype=float)
    if units in ("bohr", "atomic"):
        coords = coords * qcel.constants.bohr2angstroms

    symbols_sorted, coords_sorted = canonicalize_atom_order(list(symbols), coords, decimals=10)
    return qcel.models.Molecule(symbols=symbols_sorted, geometry=coords_sorted)


def resolve_molecule(
    calc: Calculation,
    *,
    mol_root: str | Path | None = None,
    mol_map: str | Path | None = None,
    mol_file: str | Path | None = None,
    mol_dir: str | Path | None = None,
    resolvers: list[MoleculeResolverProtocol] | None = None,
) -> Optional[qcel.models.Molecule]:
    if calc.molecule is not None:
        return calc.molecule

    if calc.code != "dalton":
        qmol = _molecule_from_calc_info(calc)
        if qmol is not None:
            calc.meta.setdefault("molecule_source", "calc_info")
            return qmol

    if calc.code != "dalton":
        root = calc.root
        preferred = root / "molecule.mol"
        if preferred.exists():
            calc.meta.setdefault("molecule_source", "calc_dir")
            calc.meta.setdefault("molecule_path", str(preferred))
            return read_mol(preferred)
        mols = list(root.glob("*.mol"))
        if mols:
            calc.meta.setdefault("molecule_source", "calc_dir")
            calc.meta.setdefault("molecule_path", str(mols[0]))
            return read_mol(mols[0])

    if mol_map is not None:
        map_path = Path(mol_map).expanduser().resolve()
        mapping = _load_map(map_path)
        label = mol_label_from_calc(calc)
        root_key = str(calc.root)
        for key in (root_key, label, getattr(calc.root, "name", None)):
            if key is None:
                continue
            if str(key) in mapping:
                mol_path = Path(mapping[str(key)]).expanduser().resolve()
                calc.meta.setdefault("molecule_source", "resolver:map")
                calc.meta.setdefault("molecule_path", str(mol_path))
                return read_mol(mol_path)

    if calc.code == "dalton" and mol_dir is not None:
        mol_dir_path = Path(mol_dir).expanduser().resolve()
        if mol_dir_path == calc.root:
            calc.meta.setdefault("warnings", []).append(
                "Refusing to use Dalton directory as molecule source; provide a separate molecule directory."
            )
            mol_dir = None

    if resolvers:
        molecule_key = calc.meta.get("molecule_key") or mol_label_from_calc(calc)
        for resolver in resolvers:
            qmol = resolver.resolve(molecule_key=molecule_key, calc=calc)
            if qmol is not None:
                return qmol

    resolver = MoleculeResolver.from_sources(
        mol_file=mol_file,
        mol_dir=mol_dir or mol_root,
        mol_map=mol_map,
    )
    res = resolver.resolve(calc)
    if res.molecule is not None:
        calc.meta.setdefault("molecule_source", f"resolver:{res.source}")
        if res.path is not None:
            calc.meta.setdefault("molecule_path", str(res.path))
        return res.molecule

    return None
