"""Tests for MADNESS parameter dataclasses and _render_value."""

from __future__ import annotations

import pytest

from gecko.workflow.params import DFTParams, MoleculeParams, ResponseParams, _render_value


class TestRenderValue:
    def test_bool_true(self):
        assert _render_value(True) == "true"

    def test_bool_false(self):
        assert _render_value(False) == "false"

    def test_int(self):
        assert _render_value(8) == "8"

    def test_float_small_scientific(self):
        assert _render_value(1e-6) == "1.0e-06"

    def test_float_zero_not_scientific(self):
        assert _render_value(0.0) == "0.0"

    def test_float_normal(self):
        assert _render_value(0.05) == "0.05"

    def test_list_of_floats(self):
        assert _render_value([0.0, 0.02, 0.04]) == "[0.0,0.02,0.04]"

    def test_list_with_small_floats(self):
        result = _render_value([1e-4, 1e-6])
        assert result == "[1.0e-04,1.0e-06]"

    def test_string(self):
        assert _render_value("b3lyp") == "b3lyp"

    def test_list_of_ints(self):
        assert _render_value([0, 1, 2]) == "[0,1,2]"


class TestDFTParams:
    def test_all_none_by_default(self):
        p = DFTParams()
        import dataclasses
        for f in dataclasses.fields(p):
            assert getattr(p, f.name) is None

    def test_set_fields(self):
        p = DFTParams(xc="b3lyp", k=10, gopt=True, print_level=5)
        assert p.xc == "b3lyp"
        assert p.k == 10
        assert p.gopt is True
        assert p.print_level == 5

    def test_protocol_list(self):
        p = DFTParams(protocol=[1e-4, 1e-6, 1e-8])
        assert p.protocol == [1e-4, 1e-6, 1e-8]


class TestMoleculeParams:
    def test_all_none_by_default(self):
        p = MoleculeParams()
        import dataclasses
        for f in dataclasses.fields(p):
            assert getattr(p, f.name) is None

    def test_set_fields(self):
        p = MoleculeParams(eprec=1e-8, no_orient=True, units="angstrom")
        assert p.eprec == 1e-8
        assert p.no_orient is True
        assert p.units == "angstrom"


class TestResponseParams:
    def test_all_none_by_default(self):
        p = ResponseParams()
        import dataclasses
        for f in dataclasses.fields(p):
            assert getattr(p, f.name) is None

    def test_set_fields(self):
        p = ResponseParams(kain=True, maxiter=40, dconv=1e-6)
        assert p.kain is True
        assert p.maxiter == 40
        assert p.dconv == 1e-6

    def test_dipole_frequencies(self):
        p = ResponseParams(dipole_frequencies=[0.0, 0.02, 0.04])
        assert p.dipole_frequencies == [0.0, 0.02, 0.04]
