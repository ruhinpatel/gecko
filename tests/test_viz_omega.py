from __future__ import annotations

import pandas as pd

from gecko.viz.omega import assign_shg_omega_index


def test_assign_shg_omega_index_starts_at_zero():
    df = pd.DataFrame(
        [
            {"molecule": "H2O", "omegaB": 0.0, "omegaC": 0.0},
            {"molecule": "H2O", "omegaB": 0.2, "omegaC": 0.2},
            {"molecule": "H2O", "omegaB": 0.4, "omegaC": 0.4},
        ]
    )

    out = assign_shg_omega_index(df, mol_col="molecule", b_col="omegaB", c_col="omegaC")
    assert out["omega"].tolist() == [0, 1, 2]
