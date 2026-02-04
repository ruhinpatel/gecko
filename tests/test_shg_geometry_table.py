from gecko.recipes.shg_csv import build_beta_table

def test_build_beta_table_with_geometry():
    df = build_beta_table(
        ["tests/fixtures/dalton_qr"],
        require_geometry=True,
        include_geometry=True,
        verbose=False,
    )

    assert not df.empty
    first = df.iloc[0]
    assert isinstance(first["geom_id"], str)
    assert isinstance(first["geometry"], str)


def test_build_beta_table_missing_geometry_allowed():
    df = build_beta_table(
        ["tests/fixtures/madness_molresponse"],
        require_geometry=False,
        verbose=False,
    )

    assert not df.empty
