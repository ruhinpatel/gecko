from __future__ import annotations

from pathlib import Path

import pandas as pd

from gecko.viz.io import load_shg_df_from_csv


def test_io_load_csv_smoke(tmp_path: Path):
    df = pd.DataFrame(
        [
            {"molecule": "H2O", "basis": "MRA", "omega": 0, "ijk": "xyz", "Beta": 1.0},
        ]
    )
    csv_path = tmp_path / "shg_ijk.csv"
    df.to_csv(csv_path, index=False)

    out = load_shg_df_from_csv(csv_path)
    assert not out.empty
    assert set(["molecule", "basis", "omega", "ijk", "Beta"]).issubset(out.columns)
