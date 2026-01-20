from __future__ import annotations

from pathlib import Path

from gecko.core.model import Calculation
from gecko.ids import calc_id, geom_id, mol_id
from gecko.molecule_id import compute_molecule_id
from gecko.molecule_resolver import resolve_molecule
from gecko.mol.resolver import mol_label_from_calc


def enrich(
    calc: Calculation,
    *,
    mol_root: str | Path | None = None,
    mol_map: str | Path | None = None,
    mol_file: str | Path | None = None,
    mol_dir: str | Path | None = None,
) -> Calculation:
    calc.meta.setdefault("label", mol_label_from_calc(calc))

    if calc.molecule is None:
        calc.molecule = resolve_molecule(
            calc,
            mol_root=mol_root,
            mol_map=mol_map,
            mol_file=mol_file,
            mol_dir=mol_dir,
        )

    calc.meta["calc_id"] = calc_id(calc)
    calc.meta["mol_id"] = mol_id(calc)

    if calc.molecule is not None:
        calc.meta["geom_id"] = geom_id(calc.molecule)
        calc.meta.setdefault("molecule_id", compute_molecule_id(calc.molecule))
        calc.meta.setdefault("mol_source", "embedded")
        calc.meta.setdefault("molecule_source", "out")
    else:
        calc.meta.setdefault("mol_source", "missing")

    basis = calc.meta.get("basis")
    if isinstance(basis, str):
        calc.meta["basis"] = " ".join(basis.split())

    return calc
