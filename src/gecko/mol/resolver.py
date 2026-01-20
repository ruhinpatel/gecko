from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Literal, Any
import json
import re

import qcelemental as qcel

from gecko.mol.io import read_mol


MolSource = Literal["embedded", "map", "dir", "file", "missing"]


@dataclass(frozen=True)
class MoleculeResolution:
    molecule: Optional[qcel.models.Molecule]
    source: MolSource
    path: Optional[Path]
    label: Optional[str]


def mol_label_from_calc(calc: Any) -> Optional[str]:
    label = None
    meta = getattr(calc, "meta", None)
    if isinstance(meta, dict) and meta.get("molecule"):
        label = str(meta.get("molecule"))
    if label is None:
        data = getattr(calc, "data", None)
        if isinstance(data, dict):
            calc_info = data.get("calc_info")
            if isinstance(calc_info, dict) and calc_info.get("molecule"):
                label = str(calc_info.get("molecule"))
    if label is None:
        root = getattr(calc, "root", None)
        if isinstance(root, Path):
            label = root.name
        elif root is not None:
            label = str(root)
    return label


def _normalize_label(label: Optional[str]) -> Optional[str]:
    if label is None:
        return None
    cleaned = str(label).strip()
    cleaned = re.sub(r"\s+", "", cleaned)
    return cleaned or None


class MoleculeResolver:
    def __init__(
        self,
        *,
        mol_file: Path | str | None = None,
        mol_dir: Path | str | None = None,
        mol_map: Path | str | None = None,
    ):
        self.mol_file = Path(mol_file).expanduser().resolve() if mol_file else None
        self.mol_dir = Path(mol_dir).expanduser().resolve() if mol_dir else None
        self.mol_map_path = Path(mol_map).expanduser().resolve() if mol_map else None
        self._mol_map: Dict[str, Path] = {}

        if self.mol_map_path is not None:
            self._mol_map = self._load_map(self.mol_map_path)

    @classmethod
    def from_sources(
        cls,
        mol_file: Path | str | None = None,
        mol_dir: Path | str | None = None,
        mol_map: Path | str | None = None,
    ) -> "MoleculeResolver":
        return cls(mol_file=mol_file, mol_dir=mol_dir, mol_map=mol_map)

    def _load_map(self, path: Path) -> Dict[str, Path]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValueError(f"Failed to read mol-map JSON: {path}") from exc

        if not isinstance(data, dict):
            raise ValueError(f"Mol-map JSON must be an object: {path}")

        mapped: Dict[str, Path] = {}
        for key, value in data.items():
            if not isinstance(value, str):
                raise ValueError(f"Mol-map entry for {key!r} must be a string path")
            norm_key = _normalize_label(str(key))
            if norm_key is None:
                continue
            mapped[norm_key] = Path(value).expanduser().resolve()
        return mapped

    def resolve(self, calc: Any) -> MoleculeResolution:
        label = mol_label_from_calc(calc)
        norm_label = _normalize_label(label)

        existing = getattr(calc, "molecule", None)
        if existing is not None:
            return MoleculeResolution(existing, "embedded", None, label)

        if norm_label is not None and norm_label in self._mol_map:
            mol_path = self._mol_map[norm_label]
            return MoleculeResolution(read_mol(mol_path), "map", mol_path, label)

        if norm_label is not None and self.mol_dir is not None:
            if getattr(calc, "code", None) == "dalton" and getattr(calc, "root", None) == self.mol_dir:
                if isinstance(getattr(calc, "meta", None), dict):
                    calc.meta.setdefault(
                        "warnings",
                        [],
                    ).append(
                        "Refusing to use Dalton directory as molecule source; provide a separate molecule directory."
                    )
                return MoleculeResolution(None, "missing", None, label)

            candidate = self.mol_dir / f"{norm_label}.mol"
            if candidate.exists():
                return MoleculeResolution(read_mol(candidate), "dir", candidate, label)

        if self.mol_file is not None:
            return MoleculeResolution(read_mol(self.mol_file), "file", self.mol_file, label)

        return MoleculeResolution(None, "missing", None, label)
