from pathlib import Path

from gecko.index import CalcIndex


def test_calc_index_groups_by_molecule_id():
    calc_dirs = [
        Path("tests/fixtures/dalton_qr"),
        Path("tests/fixtures/madness_madqc"),
    ]
    index = CalcIndex.from_dirs(calc_dirs)

    codes = {calc.code for calc in index.calcs}
    assert {"madness", "dalton"}.issubset(codes)
    assert any(calc.meta.get("molecule_id") for calc in index.calcs)
