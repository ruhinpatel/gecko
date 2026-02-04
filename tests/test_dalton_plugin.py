from pathlib import Path

import gecko
from gecko.plugins.dalton.detect import can_load, detect_dalton
from gecko.plugins.dalton.parse import split_label_basis_from_outname


FIXTURE_DIR = Path("tests/fixtures/dalton_case")
QR_FIXTURE_DIR = Path("tests/fixtures/dalton_qr")


def test_dalton_detects_outfile():
    assert can_load(FIXTURE_DIR)
    runs = detect_dalton(FIXTURE_DIR)
    assert len(runs) >= 2
    out_files = {run["meta"]["out_file"] for run in runs}
    assert "H2O-aug-cc-pVDZ.out" in out_files
    assert "nohyphen.out" in out_files
    stems = {run["meta"]["stem"] for run in runs}
    assert "H2O-aug-cc-pVDZ" in stems


def test_split_label_basis_from_outname():
    assert split_label_basis_from_outname("H2O-aug-cc-pVDZ.out") == ("H2O", "aug-cc-pVDZ")
    assert split_label_basis_from_outname("foo-bar-d-aug-cc-pCVTZ.out") == ("foo-bar", "d-aug-cc-pCVTZ")
    assert split_label_basis_from_outname("nohyphen.out") == ("nohyphen", None)


def test_dalton_parses_molecule():
    calc = gecko.load_calc(FIXTURE_DIR)
    assert calc.molecule is not None
    assert len(calc.molecule.symbols) == 3


def test_dalton_infers_basis_and_molecule():
    calc = gecko.load_calc(FIXTURE_DIR / "H2O-aug-cc-pVDZ.out")
    assert calc.meta.get("basis") == "aug-cc-pVDZ"
    inferred = calc.meta.get("inferred_from") or {}
    assert inferred.get("basis") == "filename"


def test_dalton_parses_beta_if_present():
    runs = detect_dalton(QR_FIXTURE_DIR)
    assert runs
    calc = gecko.load_calc(QR_FIXTURE_DIR)
    beta = calc.data.get("beta")
    assert beta is not None
    assert beta["omega"].shape[1] == 3
    assert len(beta["components"]) >= 1
