from pathlib import Path

from gecko.core.model import Calculation
from gecko.plugins.madness.parse import _select_input_json

import gecko


def test_parse_madqc_beta():
    calc = gecko.load_calc("tests/fixtures/madness_madqc")
    assert calc.code == "madness"
    assert calc.meta["style"] == "madqc"
    assert calc.meta.get("basis") == "MRA"

    beta = calc.data.get("beta")
    assert beta is not None
    assert beta["omega"].shape[1] == 3
    assert beta["values"].shape[1] >= 1
    assert all(len(c) == 3 and set(c) <= set("xyz") for c in beta["components"])


def test_parse_molresponse_beta():
    calc = gecko.load_calc("tests/fixtures/madness_molresponse")
    assert calc.code == "madness"
    assert calc.meta["style"] == "molresponse"
    assert calc.meta.get("basis") == "MRA"

    beta = calc.data.get("beta")
    assert beta is not None
    assert beta["omega"].shape[1] == 3
    assert beta["values"].shape[1] >= 1
    assert all(len(c) == 3 and set(c) <= set("xyz") for c in beta["components"])

def test_madqc_and_molresponse_schema_match():
    calc1 = gecko.load_calc("tests/fixtures/madness_madqc")
    calc2 = gecko.load_calc("tests/fixtures/madness_molresponse")

    b1, b2 = calc1.data["beta"], calc2.data["beta"]

    assert set(b1.keys()) == set(b2.keys())
    assert b1["omega"].shape[1] == b2["omega"].shape[1] == 3
