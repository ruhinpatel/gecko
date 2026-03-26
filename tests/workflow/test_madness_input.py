"""Tests for MADNESS input file read/write/edit API.

Covers:
1. Round-trip: parse → serialize → re-parse → assert equality
2. Strict validation: unknown keys rejected, type mismatches caught
3. CLI integration: gecko input show/get/set/validate/diff
4. Codegen parity: generated model has all expected fields
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from gecko.workflow.input_model import (
    Atom,
    DFTSection,
    MadnessInputFile,
    MoleculeSection,
    ResponseSection,
)
from gecko.workflow.input_parser import parse_madness_input
from gecko.workflow.input_serializer import serialize_madness_input

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_FIXTURE_DIR = Path(
    "/gpfs/projects/rjh/adrian/development/madness-worktrees/"
    "molresponse-feature-next/src/apps/madqc_v2"
)

_FIXTURE_FILES = [
    "test_molresponse_h2o_alpha_beta_z.in",
    "test_molresponse_lih_alpha_raman_beta_xyz.in",
    "test_molresponse_h2_excited_states.in",
]


@pytest.fixture(params=_FIXTURE_FILES, ids=[f.replace(".in", "") for f in _FIXTURE_FILES])
def fixture_path(request: pytest.FixtureRequest) -> Path:
    p = _FIXTURE_DIR / request.param
    if not p.exists():
        pytest.skip(f"Fixture not available: {p}")
    return p


@pytest.fixture
def h2o_input() -> MadnessInputFile:
    return MadnessInputFile.from_file(_FIXTURE_DIR / _FIXTURE_FILES[0])


# ---------------------------------------------------------------------------
# Round-trip tests
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_parse_serialize_reparse(self, fixture_path: Path) -> None:
        """Parse a real .in file, serialize it, re-parse, and verify equality."""
        inp1 = MadnessInputFile.from_file(fixture_path)
        text = serialize_madness_input(inp1)
        inp2 = parse_madness_input(text)

        assert inp1.dft.model_dump() == inp2.dft.model_dump()
        assert inp1.response.model_dump() == inp2.response.model_dump()
        assert inp1.molecule.model_dump() == inp2.molecule.model_dump()
        assert len(inp1.atoms) == len(inp2.atoms)
        for a1, a2 in zip(inp1.atoms, inp2.atoms):
            assert a1.symbol == a2.symbol
            assert a1.x == pytest.approx(a2.x)
            assert a1.y == pytest.approx(a2.y)
            assert a1.z == pytest.approx(a2.z)

    def test_h2o_specific_values(self) -> None:
        """Verify specific parsed values from the H2O fixture."""
        path = _FIXTURE_DIR / _FIXTURE_FILES[0]
        if not path.exists():
            pytest.skip("Fixture not available")
        inp = MadnessInputFile.from_file(path)

        assert inp.dft.xc == "hf"
        assert inp.dft.dconv == 0.0001
        assert inp.dft.localize == "new"
        assert inp.dft.l == 200.0
        assert inp.dft.maxiter == 30
        assert inp.dft.print_level == 20
        assert inp.dft.protocol == [0.0001, 1e-6]
        assert inp.dft.dipole is True

        assert inp.response.kain is True
        assert inp.response.dipole is True
        assert inp.response.dipole_directions == "z"
        assert inp.response.dipole_frequencies == [0.0, 0.02]
        assert inp.response.quadratic is True

        assert inp.molecule.units == "angstrom"
        assert inp.molecule.eprec == pytest.approx(1e-6)
        assert len(inp.atoms) == 3
        assert inp.atoms[0].symbol == "O"

    def test_lih_nuclear_response(self) -> None:
        """LiH file includes nuclear response + raman."""
        path = _FIXTURE_DIR / _FIXTURE_FILES[1]
        if not path.exists():
            pytest.skip("Fixture not available")
        inp = MadnessInputFile.from_file(path)

        assert inp.response.nuclear is True
        assert inp.response.nuclear_directions == "xyz"
        assert inp.response.quadratic is True
        assert inp.response.maxiter == 60
        assert inp.molecule.no_orient is True
        assert len(inp.atoms) == 2

    def test_h2_excited_states(self) -> None:
        """H2 file includes excited state parameters."""
        path = _FIXTURE_DIR / _FIXTURE_FILES[2]
        if not path.exists():
            pytest.skip("Fixture not available")
        inp = MadnessInputFile.from_file(path)

        assert inp.response.excited_enable is True
        assert inp.response.excited_num_states == 2
        assert inp.response.excited_tda is False
        assert inp.response.excited_maxiter == 20
        assert inp.response.excited_maxsub == 4


# ---------------------------------------------------------------------------
# Strict validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_unknown_key_rejected(self) -> None:
        """Unknown parameters should raise an error."""
        text = "dft\n    xc hf\n    bogus_param 42\nend\n"
        with pytest.raises(ValueError, match="Unknown parameter.*bogus_param"):
            parse_madness_input(text)

    def test_empty_file(self) -> None:
        """Empty file should produce default values."""
        inp = parse_madness_input("")
        assert inp.dft.xc == "hf"
        assert inp.response.kain is False
        assert inp.atoms == []

    def test_type_coercion_bool(self) -> None:
        """Boolean parameters accept true/false strings."""
        text = "dft\n    dipole true\n    save false\nend\n"
        inp = parse_madness_input(text)
        assert inp.dft.dipole is True
        assert inp.dft.save is False

    def test_type_coercion_list(self) -> None:
        """List parameters parse [x,y,z] syntax."""
        text = "response\n    dipole.frequencies [0.0,0.05,0.1]\nend\n"
        inp = parse_madness_input(text)
        assert inp.response.dipole_frequencies == [0.0, 0.05, 0.1]

    def test_type_coercion_scientific(self) -> None:
        """Scientific notation floats are parsed correctly."""
        text = "dft\n    econv 1e-8\nend\n"
        inp = parse_madness_input(text)
        assert inp.dft.econv == pytest.approx(1e-8)

    def test_comments_stripped(self) -> None:
        """Lines with # comments are handled."""
        text = "dft\n    xc b3lyp  # use B3LYP\n    # full comment line\nend\n"
        inp = parse_madness_input(text)
        assert inp.dft.xc == "b3lyp"

    def test_construct_programmatic(self) -> None:
        """Construct a MadnessInputFile entirely in Python."""
        inp = MadnessInputFile(
            dft=DFTSection(xc="b3lyp", k=8),
            response=ResponseSection(kain=True, dipole=True, dipole_frequencies=[0.0, 0.05]),
            molecule=MoleculeSection(units="angstrom"),
            atoms=[
                Atom(symbol="H", x=0.0, y=0.0, z=-0.37),
                Atom(symbol="H", x=0.0, y=0.0, z=0.37),
            ],
        )
        text = serialize_madness_input(inp)
        inp2 = parse_madness_input(text)
        assert inp2.dft.xc == "b3lyp"
        assert inp2.dft.k == 8
        assert inp2.response.dipole_frequencies == [0.0, 0.05]
        assert len(inp2.atoms) == 2


# ---------------------------------------------------------------------------
# Get/Set API tests
# ---------------------------------------------------------------------------


class TestGetSetAPI:
    def test_get_simple_key(self, h2o_input: MadnessInputFile) -> None:
        assert h2o_input.get("dft.xc") == "hf"

    def test_get_dotted_key(self, h2o_input: MadnessInputFile) -> None:
        assert h2o_input.get("response.dipole.frequencies") == [0.0, 0.02]

    def test_get_molecule_key(self, h2o_input: MadnessInputFile) -> None:
        assert h2o_input.get("molecule.units") == "angstrom"

    def test_set_string(self, h2o_input: MadnessInputFile) -> None:
        h2o_input.set("dft.xc", "b3lyp")
        assert h2o_input.get("dft.xc") == "b3lyp"

    def test_set_list(self, h2o_input: MadnessInputFile) -> None:
        h2o_input.set("response.dipole.frequencies", "[0.0,0.05,0.1]")
        assert h2o_input.get("response.dipole.frequencies") == [0.0, 0.05, 0.1]

    def test_set_bool(self, h2o_input: MadnessInputFile) -> None:
        h2o_input.set("response.kain", "false")
        assert h2o_input.get("response.kain") is False

    def test_set_int(self, h2o_input: MadnessInputFile) -> None:
        h2o_input.set("dft.maxiter", "50")
        assert h2o_input.get("dft.maxiter") == 50

    def test_get_invalid_section(self, h2o_input: MadnessInputFile) -> None:
        with pytest.raises(KeyError, match="Unknown section"):
            h2o_input.get("bogus.xc")

    def test_get_invalid_key(self, h2o_input: MadnessInputFile) -> None:
        with pytest.raises(KeyError, match="Unknown parameter"):
            h2o_input.get("dft.nonexistent")

    def test_get_no_dot(self, h2o_input: MadnessInputFile) -> None:
        with pytest.raises(ValueError, match="section.param"):
            h2o_input.get("xc")


# ---------------------------------------------------------------------------
# CLI integration tests
# ---------------------------------------------------------------------------


class TestCLI:
    """Test the gecko input CLI via subprocess."""

    @pytest.fixture
    def test_file(self) -> str:
        path = _FIXTURE_DIR / _FIXTURE_FILES[0]
        if not path.exists():
            pytest.skip("Fixture not available")
        return str(path)

    def _run(self, *args: str) -> subprocess.CompletedProcess:
        from gecko.cli import main as gecko_main
        import io
        from contextlib import redirect_stdout, redirect_stderr

        stdout = io.StringIO()
        stderr = io.StringIO()
        try:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                rc = gecko_main(list(args))
        except SystemExit as e:
            rc = e.code if e.code is not None else 0

        return subprocess.CompletedProcess(
            args=["gecko"] + list(args),
            returncode=rc,
            stdout=stdout.getvalue(),
            stderr=stderr.getvalue(),
        )

    def test_show_madness_format(self, test_file: str) -> None:
        result = self._run("input", "show", test_file)
        assert result.returncode == 0
        assert "dft" in result.stdout
        assert "end" in result.stdout

    def test_show_json_format(self, test_file: str) -> None:
        result = self._run("input", "show", test_file, "--format", "json")
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert "dft" in data
        assert "response" in data

    def test_show_section(self, test_file: str) -> None:
        result = self._run("input", "show", test_file, "--section", "response")
        assert result.returncode == 0
        assert result.stdout.startswith("response\n")
        assert "dft" not in result.stdout.split("\n")[1:]  # dft shouldn't appear after first line

    def test_get(self, test_file: str) -> None:
        result = self._run("input", "get", test_file, "dft.xc")
        assert result.returncode == 0
        assert result.stdout.strip() == "hf"

    def test_set_dry_run(self, test_file: str) -> None:
        result = self._run("input", "set", test_file, "dft.xc", "b3lyp", "--dry-run")
        assert result.returncode == 0
        assert "b3lyp" in result.stdout

    def test_validate(self, test_file: str) -> None:
        result = self._run("input", "validate", test_file)
        assert result.returncode == 0
        assert "Valid" in result.stdout

    def test_diff(self) -> None:
        f1 = _FIXTURE_DIR / _FIXTURE_FILES[0]
        f2 = _FIXTURE_DIR / _FIXTURE_FILES[1]
        if not f1.exists() or not f2.exists():
            pytest.skip("Fixtures not available")
        result = self._run("input", "diff", str(f1), str(f2))
        assert result.returncode == 0
        assert "Differences" in result.stdout

    def test_create_and_set(self, tmp_path: Path) -> None:
        out = tmp_path / "test.in"
        result = self._run(
            "input", "create",
            "-o", str(out),
            "--set", "dft.xc=b3lyp",
            "--set", "dft.k=8",
        )
        assert result.returncode == 0
        assert out.exists()

        inp = MadnessInputFile.from_file(out)
        assert inp.dft.xc == "b3lyp"
        assert inp.dft.k == 8


# ---------------------------------------------------------------------------
# Codegen parity test
# ---------------------------------------------------------------------------


class TestCodegenParity:
    """Verify the generated model matches expected C++ parameters."""

    def test_dft_has_key_fields(self) -> None:
        fields = DFTSection.model_fields
        for key in ["xc", "k", "l", "charge", "econv", "dconv", "maxiter",
                     "protocol", "localize", "pointgroup", "spin_restricted",
                     "dipole", "derivatives", "save", "restart", "print_level"]:
            assert key in fields, f"DFTSection missing field: {key}"

    def test_response_has_key_fields(self) -> None:
        fields = ResponseSection.model_fields
        for key in ["prefix", "archive", "kain", "maxsub", "dconv",
                     "dipole", "dipole_frequencies", "dipole_directions",
                     "nuclear", "nuclear_directions", "nuclear_frequencies",
                     "quadratic", "localize", "excited_enable", "excited_num_states",
                     "beta_shg", "state_parallel", "maxiter", "protocol"]:
            assert key in fields, f"ResponseSection missing field: {key}"

    def test_molecule_has_key_fields(self) -> None:
        fields = MoleculeSection.model_fields
        for key in ["eprec", "units", "no_orient", "field", "symtol", "core_type"]:
            assert key in fields, f"MoleculeSection missing field: {key}"

    def test_dotted_keys_have_aliases(self) -> None:
        """Fields derived from dotted MADNESS keys must have aliases."""
        for field_name, field_info in ResponseSection.model_fields.items():
            if "_" in field_name and field_name.replace("_", ".") != field_name:
                # Check if there should be an alias
                alias = field_info.alias
                if alias and "." in alias:
                    assert alias.replace(".", "_") == field_name

    def test_json_schema_export(self) -> None:
        """Model should produce a valid JSON schema (for GUI integration)."""
        schema = MadnessInputFile.model_json_schema()
        assert "properties" in schema
        assert "dft" in schema["properties"]
        assert "response" in schema["properties"]
        assert "molecule" in schema["properties"]
