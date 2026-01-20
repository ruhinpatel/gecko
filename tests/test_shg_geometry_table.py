from pathlib import Path

from gecko.recipes.shg_csv import build_beta_table


FIXTURES = Path(__file__).resolve().parent / "fixtures"


def _fixture_path(*parts: str) -> str:
    return str((FIXTURES / Path(*parts)).resolve())


def test_build_beta_table_with_geometry():
    df = build_beta_table(
        ["tests/fixtures/calc_missing_geom"],
        require_geometry=True,
        include_geometry=True,
        mol_dir=_fixture_path("mols_dir"),
        verbose=False,
    )

    assert not df.empty
    first = df.iloc[0]
    assert isinstance(first["geom_id"], str)
    assert isinstance(first["geometry"], str)


def test_build_beta_table_missing_geometry_allowed():
    df = build_beta_table(
        ["tests/fixtures/calc_missing_geom"],
        require_geometry=False,
        verbose=False,
    )

    assert not df.empty
