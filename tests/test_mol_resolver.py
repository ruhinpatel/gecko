from pathlib import Path

import qcelemental as qcel

import gecko
from gecko.core.model import Calculation
from gecko.mol.resolver import MoleculeResolver


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_path(*parts: str) -> Path:
    return (FIXTURES / Path(*parts)).resolve()


def test_embedded_wins():
    qmol = qcel.models.Molecule(symbols=["H"], geometry=[[0.0, 0.0, 0.0]])
    calc = Calculation(code="madness", root=Path("H2O"), data={}, meta={})
    calc.molecule = qmol

    resolver = MoleculeResolver.from_sources(
        mol_file=_fixture_path("mols", "H2O.mol"),
        mol_dir=_fixture_path("mols_dir"),
        mol_map=_fixture_path("mol_map.json"),
    )

    res = resolver.resolve(calc)
    assert res.source == "embedded"
    assert res.molecule is qmol


def test_map_beats_dir():
    resolver = MoleculeResolver.from_sources(
        mol_map=_fixture_path("mol_map.json"),
        mol_dir=_fixture_path("mols_dir"),
    )
    calc = Calculation(code="madness", root=Path("H2O"), data={}, meta={"molecule": "H2O"})

    res = resolver.resolve(calc)
    assert res.source == "map"
    assert res.path == _fixture_path("mols", "H2O.mol")


def test_dir_beats_file():
    resolver = MoleculeResolver.from_sources(
        mol_dir=_fixture_path("mols_dir"),
        mol_file=_fixture_path("mols", "H2O.mol"),
    )
    calc = Calculation(code="madness", root=Path("H2O"), data={}, meta={"molecule": "H2O"})

    res = resolver.resolve(calc)
    assert res.source == "dir"
    assert res.path == _fixture_path("mols_dir", "H2O.mol")


def test_missing_is_safe():
    resolver = MoleculeResolver.from_sources()
    calc = gecko.load_calc("tests/fixtures/calc_missing_geom", mol_resolver=resolver)

    assert calc.molecule is None
    assert calc.meta.get("mol_source") == "missing"
