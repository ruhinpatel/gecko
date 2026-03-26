"""Tests for geometry acquisition utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qcelemental as qcel

from gecko.workflow.geometry import load_geometry_from_file, _atomic_number_to_symbol


class TestAtomicNumberToSymbol:
    def test_common_elements(self):
        assert _atomic_number_to_symbol(1) == "H"
        assert _atomic_number_to_symbol(8) == "O"
        assert _atomic_number_to_symbol(6) == "C"
        assert _atomic_number_to_symbol(16) == "S"
        assert _atomic_number_to_symbol(7) == "N"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Atomic number"):
            _atomic_number_to_symbol(200)


class TestLoadGeometryFromFile:
    def test_load_xyz(self, tmp_path):
        xyz = tmp_path / "water.xyz"
        xyz.write_text(
            "3\n"
            "water\n"
            "O  0.000  0.000  0.117\n"
            "H  0.000  0.757 -0.469\n"
            "H  0.000 -0.757 -0.469\n"
        )
        mol = load_geometry_from_file(xyz)
        assert list(mol.symbols) == ["O", "H", "H"]
        # geometry is (N, 3) in qcel
        assert np.asarray(mol.geometry).size == 9

    def test_load_xyz_coordinates_in_bohr(self, tmp_path):
        """XYZ is Angstrom → should be converted to Bohr internally."""
        xyz = tmp_path / "h2.xyz"
        xyz.write_text("2\nh2\nH  0.0  0.0  0.0\nH  0.0  0.0  0.74\n")
        mol = load_geometry_from_file(xyz)
        ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")
        geom = np.asarray(mol.geometry).reshape(-1, 3)
        dist_bohr = abs(geom[1, 2] - geom[0, 2])  # z-difference between H atoms
        assert abs(dist_bohr - 0.74 * ang2bohr) < 1e-6

    def test_unsupported_extension_raises(self, tmp_path):
        bad = tmp_path / "mol.pdb"
        bad.write_text("")
        with pytest.raises(ValueError, match="Unsupported file extension"):
            load_geometry_from_file(bad)

    def test_load_madness_mol(self, tmp_path):
        mol_file = tmp_path / "h2o.mol"
        mol_file.write_text(
            "molecule\n"
            "  eprec 1e-4\n"
            "  units atomic\n"
            "  O  0.0  0.0  0.2129\n"
            "  H  0.0  1.4213 -0.8517\n"
            "  H  0.0 -1.4213 -0.8517\n"
            "end\n"
        )
        mol = load_geometry_from_file(mol_file)
        assert len(mol.symbols) == 3
        assert mol.symbols[0] == "O"


class TestFetchGeometry:
    """Network-dependent tests; skipped in offline environments."""

    def test_unknown_source_raises(self):
        from gecko.workflow.geometry import fetch_geometry
        with pytest.raises(ValueError, match="Unknown source"):
            fetch_geometry("water", source="nonexistent")
