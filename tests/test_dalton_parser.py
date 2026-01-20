from pathlib import Path

import pytest

import gecko


def test_parse_mol_block_geometry():
    out_path = Path("tests/fixtures/dalton/mol_content_block/example.out")
    calc = gecko.load_calc(out_path)

    assert calc.molecule is not None
    assert len(calc.molecule.symbols) == 14
    symbols = list(calc.molecule.symbols)
    assert symbols[:4] == ["C", "C", "C", "C"]
    assert symbols[4:6] == ["N", "N"]
    assert symbols[6:8] == ["O", "O"]
    assert calc.molecule.geometry.shape == (14, 3)
    assert calc.meta.get("basis") == "aug-cc-pVDZ"
    assert calc.meta.get("molecule_id") is not None


def test_load_calcs_multi_out_dir():
    root = Path("tests/fixtures/dalton/multi_out_dir")
    calcs = gecko.load_calcs(root)
    assert len(calcs) == 2


def test_load_calc_multi_out_dir_requires_file():
    root = Path("tests/fixtures/dalton/multi_out_dir")
    with pytest.raises(ValueError, match="Multiple Dalton .out files"):
        gecko.load_calc(root)
