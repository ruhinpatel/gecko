"""Tests for workflow input file writers."""

from __future__ import annotations

import numpy as np
import pytest
import qcelemental as qcel

from gecko.workflow.writers import MadnessInput, DaltonInput, generate_calc_dir


@pytest.fixture()
def water() -> qcel.models.Molecule:
    """Water molecule at a known geometry (Bohr)."""
    ang2bohr = qcel.constants.conversion_factor("angstrom", "bohr")
    coords_ang = np.array([
        [0.0000,  0.0000,  0.1173],
        [0.0000,  0.7572, -0.4692],
        [0.0000, -0.7572, -0.4692],
    ])
    return qcel.models.Molecule(
        symbols=["O", "H", "H"],
        geometry=(coords_ang * ang2bohr).flatten().tolist(),
        name="H2O",
    )


# ---------------------------------------------------------------------------
# MadnessInput
# ---------------------------------------------------------------------------

class TestMadnessInput:
    def test_writes_in_file(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", property="alpha")
        path = inp.write(tmp_path)
        assert path.exists()
        assert path.suffix == ".in"
        assert path.name == "alpha_h2o.in"

    def test_dft_section_present(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", xc="hf")
        content = path = inp.write(tmp_path).read_text()
        assert "dft" in content
        assert "end" in content

    def test_molecule_section_has_atoms(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O")
        content = inp.write(tmp_path).read_text()
        assert "molecule" in content
        assert "units  atomic" in content
        assert "O " in content
        assert "H " in content

    def test_response_alpha_keywords(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", property="alpha")
        content = inp.write(tmp_path).read_text()
        assert "response" in content
        assert "polarizability" in content

    def test_response_beta_keywords(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", property="beta")
        content = inp.write(tmp_path).read_text()
        assert "quadratic true" in content
        assert "hyperpolarizability" in content

    def test_response_raman_keywords(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", property="raman")
        content = inp.write(tmp_path).read_text()
        assert "nuclear true" in content
        assert "raman" in content

    def test_custom_frequencies(self, tmp_path, water):
        inp = MadnessInput(
            molecule=water, mol_name="H2O", property="alpha",
            frequencies=[0.0, 0.05, 0.1],
        )
        content = inp.write(tmp_path).read_text()
        assert "0.0,0.05,0.1" in content

    def test_dft_xc_written_when_not_hf(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", xc="b3lyp")
        content = inp.write(tmp_path).read_text()
        assert "xc b3lyp" in content

    def test_hf_xc_not_written(self, tmp_path, water):
        inp = MadnessInput(molecule=water, mol_name="H2O", xc="hf")
        content = inp.write(tmp_path).read_text()
        assert "xc hf" not in content


# ---------------------------------------------------------------------------
# DaltonInput
# ---------------------------------------------------------------------------

class TestDaltonInput:
    def test_writes_dal_and_mol(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ")
        paths = inp.write(tmp_path)
        assert paths["dal"].exists()
        assert paths["mol"].exists()

    def test_dal_has_dalton_header(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ")
        content = inp.write(tmp_path)["dal"].read_text()
        assert "**DALTON INPUT" in content
        assert "**END OF DALTON INPUT" in content

    def test_dal_alpha_keywords(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ", property="alpha")
        content = inp.write(tmp_path)["dal"].read_text()
        assert "*LINEAR" in content
        assert ".ALPHA" in content

    def test_dal_beta_keywords(self, tmp_path, water):
        inp = DaltonInput(
            molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
            property="beta", frequencies=[0.0, 0.05],
        )
        content = inp.write(tmp_path)["dal"].read_text()
        assert "*QUADRA" in content
        assert ".DIPLEN" in content
        assert ".FREQUENCIES" in content

    def test_mol_contains_basis(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ")
        content = inp.write(tmp_path)["mol"].read_text()
        assert "aug-cc-pVDZ" in content

    def test_mol_contains_atoms(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ")
        content = inp.write(tmp_path)["mol"].read_text()
        assert "Charge=8" in content   # oxygen
        assert "Charge=1" in content   # hydrogen

    def test_hf_wave_function(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ", xc="hf")
        content = inp.write(tmp_path)["dal"].read_text()
        assert ".HF" in content

    def test_dft_wave_function(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ", xc="b3lyp")
        content = inp.write(tmp_path)["dal"].read_text()
        assert ".DFT" in content
        assert "B3LYP" in content


# ---------------------------------------------------------------------------
# generate_calc_dir
# ---------------------------------------------------------------------------

class TestGenerateCalcDir:
    def test_madness_only(self, tmp_path, water):
        result = generate_calc_dir(
            molecule=water,
            mol_name="H2O",
            property="alpha",
            codes=["madness"],
            basis_sets=["aug-cc-pVDZ"],
            out_dir=tmp_path,
        )
        assert len(result["madness"]) == 1
        assert result["madness"][0].suffix == ".in"
        assert len(result["dalton"]) == 0

    def test_dalton_multiple_bases(self, tmp_path, water):
        result = generate_calc_dir(
            molecule=water,
            mol_name="H2O",
            property="alpha",
            codes=["dalton"],
            basis_sets=["aug-cc-pVDZ", "aug-cc-pVTZ"],
            out_dir=tmp_path,
        )
        assert len(result["madness"]) == 0
        dal_files = [p for p in result["dalton"] if p.suffix == ".dal"]
        assert len(dal_files) == 2

    def test_both_codes(self, tmp_path, water):
        result = generate_calc_dir(
            molecule=water,
            mol_name="H2O",
            property="alpha",
            codes=["madness", "dalton"],
            basis_sets=["aug-cc-pVDZ"],
            out_dir=tmp_path,
        )
        assert len(result["madness"]) == 1
        assert len(result["dalton"]) > 0

    def test_madness_tier_directory(self, tmp_path, water):
        result = generate_calc_dir(
            molecule=water,
            mol_name="H2O",
            property="alpha",
            codes=["madness"],
            basis_sets=[],
            out_dir=tmp_path,
            tier="medium",
        )
        in_file = result["madness"][0]
        assert in_file.parent.name == "mad-medium"
        assert in_file.parent.parent.name == "H2O"

    def test_madness_no_tier_uses_madness_dir(self, tmp_path, water):
        result = generate_calc_dir(
            molecule=water,
            mol_name="H2O",
            property="alpha",
            codes=["madness"],
            basis_sets=[],
            out_dir=tmp_path,
        )
        assert result["madness"][0].parent.name == "madness"


# ---------------------------------------------------------------------------
# MadnessInput — params overrides
# ---------------------------------------------------------------------------


class TestMadnessInputOverrides:
    def test_dft_params_override_k(self, tmp_path, water):
        from gecko.workflow.params import DFTParams
        inp = MadnessInput(molecule=water, mol_name="H2O",
                           dft_params=DFTParams(k=10))
        content = inp.write(tmp_path).read_text()
        assert "k 10" in content

    def test_dft_params_gopt(self, tmp_path, water):
        from gecko.workflow.params import DFTParams
        inp = MadnessInput(molecule=water, mol_name="H2O",
                           dft_params=DFTParams(gopt=True, print_level=5))
        content = inp.write(tmp_path).read_text()
        assert "gopt true" in content
        assert "print_level 5" in content

    def test_dft_params_hf_xc_still_omitted(self, tmp_path, water):
        from gecko.workflow.params import DFTParams
        inp = MadnessInput(molecule=water, mol_name="H2O",
                           dft_params=DFTParams(xc="hf"))
        content = inp.write(tmp_path).read_text()
        assert "xc hf" not in content

    def test_dft_params_dft_xc_written(self, tmp_path, water):
        from gecko.workflow.params import DFTParams
        inp = MadnessInput(molecule=water, mol_name="H2O",
                           dft_params=DFTParams(xc="b3lyp"))
        content = inp.write(tmp_path).read_text()
        assert "xc b3lyp" in content

    def test_molecule_params_eprec(self, tmp_path, water):
        from gecko.workflow.params import MoleculeParams
        inp = MadnessInput(molecule=water, mol_name="H2O",
                           molecule_params=MoleculeParams(eprec=1e-8, no_orient=True))
        content = inp.write(tmp_path).read_text()
        assert "eprec" in content
        assert "no_orient  true" in content

    def test_response_params_kain_and_maxiter(self, tmp_path, water):
        from gecko.workflow.params import ResponseParams
        inp = MadnessInput(molecule=water, mol_name="H2O", property="alpha",
                           response_params=ResponseParams(kain=True, maxiter=50))
        content = inp.write(tmp_path).read_text()
        assert "kain true" in content
        assert "maxiter 50" in content

    def test_response_params_override_requested_properties(self, tmp_path, water):
        from gecko.workflow.params import ResponseParams
        inp = MadnessInput(molecule=water, mol_name="H2O", property="alpha",
                           response_params=ResponseParams(
                               requested_properties=["polarizability", "oscillator_strength"]
                           ))
        content = inp.write(tmp_path).read_text()
        assert "oscillator_strength" in content

    def test_no_params_gives_same_output(self, tmp_path, water):
        """Regression: no params args → output identical to original code."""
        inp = MadnessInput(molecule=water, mol_name="H2O", property="alpha")
        content = inp.write(tmp_path).read_text()
        assert "polarizability" in content
        assert "k 8" in content
        assert "econv 1.0e-06" in content


# ---------------------------------------------------------------------------
# DaltonInput — corrected Raman (two-file .WALK workflow)
# ---------------------------------------------------------------------------


class TestDaltonRaman:
    def test_raman_produces_three_files(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="raman", frequencies=[0.0, 0.02, 0.04])
        files = inp.write(tmp_path)
        assert "optimize_dal" in files
        assert "raman_dal" in files
        assert "mol" in files
        assert files["optimize_dal"].exists()
        assert files["raman_dal"].exists()
        assert files["mol"].exists()

    def test_optimize_dal_has_optimize_keyword(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="raman")
        files = inp.write(tmp_path)
        content = files["optimize_dal"].read_text()
        assert ".OPTIMIZE" in content
        assert ".2NDORD" in content
        assert ".VIBANA" in content

    def test_raman_dal_uses_walk(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="raman", frequencies=[0.0, 0.02, 0.04])
        content = inp.write(tmp_path)["raman_dal"].read_text()
        assert ".WALK" in content
        assert ".NUMERI" in content

    def test_raman_frequencies_in_all_three_blocks(self, tmp_path, water):
        freqs = [0.0, 0.02, 0.04, 0.06, 0.08, 0.1]
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="raman", frequencies=freqs)
        content = inp.write(tmp_path)["raman_dal"].read_text()
        freq_line = " ".join(str(f) for f in freqs)
        assert content.count(freq_line) == 3
        assert content.count(".FREQUE") == 3
        assert content.count("6") >= 3  # frequency count appears 3 times

    def test_raman_has_start_each_step_properties_sections(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="raman")
        content = inp.write(tmp_path)["raman_dal"].read_text()
        assert "**START" in content
        assert "**EACH STEP" in content
        assert "**PROPERTIES" in content

    def test_alpha_still_produces_two_files(self, tmp_path, water):
        inp = DaltonInput(molecule=water, mol_name="H2O", basis="aug-cc-pVDZ",
                          property="alpha")
        files = inp.write(tmp_path)
        assert "dal" in files
        assert "mol" in files
        assert "optimize_dal" not in files
