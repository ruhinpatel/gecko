from pathlib import Path

import qcelemental as qcel

import gecko
from gecko.molecule_id import compute_molecule_id


def test_compute_molecule_id_is_stable():
    mol = qcel.models.Molecule(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    assert compute_molecule_id(mol) == compute_molecule_id(mol)


def test_compute_molecule_id_is_strict():
    mol = qcel.models.Molecule(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    shifted = qcel.models.Molecule(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74 + 1e-12]])
    assert compute_molecule_id(mol) != compute_molecule_id(shifted)


def test_madness_input_fallback_sets_molecule_id():
    fixture = Path("tests/fixtures/madness_missing_molecule")
    calc = gecko.load_calc(fixture)
    assert calc.molecule is not None
    assert calc.meta.get("molecule_id")
