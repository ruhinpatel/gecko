from __future__ import annotations

import numpy as np
import qcelemental as qcel

from gecko.molecule.canonical import canonicalize_atom_order


def _mol_hash(mol: qcel.models.Molecule) -> str:
    if hasattr(mol, "get_hash"):
        return mol.get_hash()
    return mol.hash  # type: ignore[attr-defined]


def _water_symbols_coords():
    symbols = ["O", "H", "H"]
    coords = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=float,
    )
    return symbols, coords


def test_hash_mismatch_without_canonicalization():
    symbols, coords = _water_symbols_coords()
    perm = [1, 0, 2]
    symbols2 = [symbols[i] for i in perm]
    coords2 = coords[perm]

    mol1 = qcel.models.Molecule(symbols=symbols, geometry=coords)
    mol2 = qcel.models.Molecule(symbols=symbols2, geometry=coords2)

    assert _mol_hash(mol1) != _mol_hash(mol2)


def test_hash_match_with_canonicalization():
    symbols, coords = _water_symbols_coords()
    perm = [1, 0, 2]
    symbols2 = [symbols[i] for i in perm]
    coords2 = coords[perm]

    s1, c1 = canonicalize_atom_order(symbols, coords, decimals=10)
    s2, c2 = canonicalize_atom_order(symbols2, coords2, decimals=10)

    mol1 = qcel.models.Molecule(symbols=s1, geometry=c1)
    mol2 = qcel.models.Molecule(symbols=s2, geometry=c2)

    assert _mol_hash(mol1) == _mol_hash(mol2)


def test_rounding_stability():
    symbols, coords = _water_symbols_coords()
    perm = [1, 0, 2]
    symbols2 = [symbols[i] for i in perm]
    coords2 = coords[perm].copy()
    coords2[0, 0] += 1e-12

    s1, c1 = canonicalize_atom_order(symbols, coords, decimals=10)
    s2, c2 = canonicalize_atom_order(symbols2, coords2, decimals=10)

    mol1 = qcel.models.Molecule(symbols=s1, geometry=c1)
    mol2 = qcel.models.Molecule(symbols=s2, geometry=c2)

    assert _mol_hash(mol1) == _mol_hash(mol2)
