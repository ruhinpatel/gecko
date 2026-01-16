import gecko

def test_beta_component_normalization_molresponse_fixture():
    calc = gecko.load_calc("tests/fixtures/madness_molresponse")
    beta = calc.data["beta"]
    assert beta["values"].shape[1] == 27
    assert set(beta["components"]) == {
        a+b+c for a in "xyz" for b in "xyz" for c in "xyz"
    }
