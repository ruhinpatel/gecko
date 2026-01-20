from __future__ import annotations

from pathlib import Path

import pandas as pd

import gecko
from gecko.tables.shg import assign_shg_omega_index, build_shg_ijk, filter_shg_rows


def test_assign_shg_omega_index_per_molecule():
    df = pd.DataFrame(
        [
            {"geom_id": "g1", "mol_id": "H2O", "root": "r1", "omegaB": 0.0, "omegaC": 0.0},
            {"geom_id": "g1", "mol_id": "H2O", "root": "r1", "omegaB": 0.1, "omegaC": 0.1},
            {"geom_id": "g2", "mol_id": "H2O2", "root": "r2", "omegaB": 0.0, "omegaC": 0.0},
            {"geom_id": None, "mol_id": "CO2", "root": "r3", "omegaB": 0.2, "omegaC": 0.2},
            {"geom_id": None, "mol_id": "CO2", "root": "r3", "omegaB": 0.4, "omegaC": 0.4},
        ]
    )

    out = assign_shg_omega_index(df, start_at=0)

    g1 = out[out["geom_id"] == "g1"].sort_values("omegaB")
    assert g1["omega"].tolist() == [0, 1]

    g2 = out[out["geom_id"] == "g2"]
    assert g2["omega"].tolist() == [0]

    co2 = out[out["mol_id"] == "CO2"].sort_values("omegaB")
    assert co2["omega"].tolist() == [0, 1]


def test_filter_shg_rows_tolerance():
    df = pd.DataFrame(
        [
            {"omegaB": 0.0, "omegaC": 0.0},
            {"omegaB": 0.1, "omegaC": 0.1 + 1e-13},
            {"omegaB": 0.2, "omegaC": 0.2001},
        ]
    )
    out = filter_shg_rows(df, tol=1e-12)
    assert len(out) == 2


def test_build_shg_ijk_mixed_code_fixture():
    dalton_path = Path(
        "tests/fixtures/calc_nlo_beta/NLO/dalton/hf/n2/dipole/quad_n2-aug-cc-pVDZ.out"
    )
    madness_path = Path("tests/fixtures/calc_nlo_beta/NLO/madness/n2/output.json")

    calc_dalton = gecko.load_calc(dalton_path)
    calc_madness = gecko.load_calc(madness_path)
    assert calc_dalton.molecule is not None
    assert calc_madness.molecule is not None
    assert calc_dalton.molecule.get_hash() == calc_madness.molecule.get_hash()

    shg_df = build_shg_ijk([dalton_path, madness_path], start_at=0, tol=1e-12)
    assert set(shg_df["code"].unique()) == {"dalton", "madness"}

    madness_basis = shg_df[shg_df["code"] == "madness"]["basis"].unique().tolist()
    assert madness_basis == ["MRA"]

    dalton_basis = shg_df[shg_df["code"] == "dalton"]["basis"].unique().tolist()
    assert all(b != "MRA" for b in dalton_basis)


def test_shg_omega_index_notebook_parity():
    df = pd.DataFrame(
        [
            {"mol_id": "H2O", "root": "r1", "omegaB": 0.0, "omegaC": 0.0},
            {"mol_id": "H2O", "root": "r1", "omegaB": 0.0, "omegaC": 0.0},
            {"mol_id": "H2O", "root": "r1", "omegaB": 0.2, "omegaC": 0.2},
            {"mol_id": "CO2", "root": "r2", "omegaB": 0.1, "omegaC": 0.1},
        ]
    )
    out = assign_shg_omega_index(df, start_at=0)

    h2o = out[out["mol_id"] == "H2O"].sort_values("omegaB")
    assert h2o["omega"].tolist() == [0, 0, 1]

    co2 = out[out["mol_id"] == "CO2"]
    assert co2["omega"].tolist() == [0]
