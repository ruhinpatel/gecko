from collections import defaultdict
from pathlib import Path

from gecko.core.iterators import iter_calc_dirs
from gecko.index import CalcIndex


def test_calc_index_groups_by_molecule_id():
    root = Path("tests/fixtures/calc_nlo_beta/NLO")
    calc_dirs = list(iter_calc_dirs(root))
    index = CalcIndex.from_dirs(calc_dirs)

    codes_by_molecule = defaultdict(set)
    for calc in index.calcs:
        mol_id = calc.meta.get("molecule_id")
        if mol_id:
            codes_by_molecule[mol_id].add(calc.code)

    assert any(codes >= {"madness", "dalton"} for codes in codes_by_molecule.values())
