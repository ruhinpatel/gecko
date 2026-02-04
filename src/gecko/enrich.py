from __future__ import annotations

from pathlib import Path

from gecko.core.model import Calculation
from gecko.ids import calc_id, geom_id_from_molecule, mol_id_from_molecule


def _label_from_calc(calc: Calculation) -> str:
    meta = calc.meta or {}
    for key in ("label", "molecule", "molecule_key", "run_id"):
        val = meta.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return calc.root.name


def enrich(
    calc: Calculation,
) -> Calculation:
    calc.meta.setdefault("label", _label_from_calc(calc))

    if calc.molecule is None:
        if calc.data.get("molecule") is not None:
            calc.molecule = calc.data.get("molecule")
        else:
            from gecko.mol.io import read_mol

            preferred = calc.root / "molecule.mol"
            candidates = [preferred] if preferred.exists() else []
            candidates.extend(sorted(p for p in calc.root.glob("*.mol") if p != preferred))
            for path in candidates:
                try:
                    calc.molecule = read_mol(path)
                    calc.meta.setdefault("molecule_source", "calc_dir")
                    calc.meta.setdefault("molecule_path", str(path))
                    break
                except Exception as exc:
                    calc.meta.setdefault("warnings", []).append(
                        f"Failed to read molecule .mol file: {path} ({type(exc).__name__}: {exc})"
                    )

    calc.meta["calc_id"] = calc_id(calc)
    calc.meta["mol_id"] = mol_id_from_molecule(calc.molecule)

    if calc.molecule is not None:
        calc.meta["geom_id"] = geom_id_from_molecule(calc.molecule)
        calc.meta["molecule_id"] = calc.meta.get("geom_id")
        calc.meta.setdefault("mol_source", "embedded")
    else:
        calc.meta.setdefault("mol_source", "missing")

    basis = calc.meta.get("basis")
    if isinstance(basis, str):
        calc.meta["basis"] = " ".join(basis.split())

    return calc
