from __future__ import annotations

import numpy as np
import qcelemental as qcel

from gecko.ids import geom_id_from_molecule, mol_id_from_molecule


def test_geom_id_matches_qcel_hash():
    mol = qcel.models.Molecule(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    assert geom_id_from_molecule(mol) == mol.get_hash()


def test_mol_id_matches_qcel_formula():
    mol = qcel.models.Molecule(
        symbols=["O", "H", "H"],
        geometry=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
            ]
        ),
    )
    if hasattr(mol, "formula"):
        expected = str(mol.formula)
    elif hasattr(mol, "get_molecular_formula"):
        expected = str(mol.get_molecular_formula())
    else:
        expected = str(qcel.molutil.molecular_formula(mol.symbols))

    assert mol_id_from_molecule(mol) == expected


def test_ids_handle_none_molecule():
    assert geom_id_from_molecule(None) is None
    assert mol_id_from_molecule(None) is None
