from pathlib import Path

import gecko


def test_parse_mol_block_geometry():
    out_path = Path("tests/fixtures/dalton/mol_content_block/example.out")
    calc = gecko.load_calc(out_path)

    assert calc.molecule is not None
    assert len(calc.molecule.symbols) == 14
    symbols = list(calc.molecule.symbols)
    # Molecule atom order is canonicalized for stable hashing (H first).
    assert symbols[:6] == ["H", "H", "H", "H", "H", "H"]
    assert symbols[6:10] == ["C", "C", "C", "C"]
    assert symbols[10:12] == ["N", "N"]
    assert symbols[12:14] == ["O", "O"]
    assert calc.molecule.geometry.shape == (14, 3)
    assert calc.meta.get("basis") == "aug-cc-pVDZ"
    assert calc.meta.get("molecule_id") is not None


def test_load_calcs_multi_out_dir():
    root = Path("tests/fixtures/dalton/multi_out_dir")
    calcs = gecko.load_calcs(root)
    assert len(calcs) == 1
    assert calcs[0].code == "dalton"


def test_load_calc_multi_out_dir_collects_outputs():
    root = Path("tests/fixtures/dalton/multi_out_dir")
    calc = gecko.load_calc(root)
    outputs = calc.data.get("dalton_outputs") or {}
    assert isinstance(outputs, dict)
    assert len(outputs) >= 2
