from __future__ import annotations

from typing import Dict

import numpy as np

from migration.viz.field_error import (
    ErrorSettings,
    compute_error_fields,
    evaluate_field,
    load_lebedev_grid,
)


def _beta_df_to_np(beta_df) -> np.ndarray:
    ijk_map = {"X": 0, "Y": 1, "Z": 2}
    beta_np = np.zeros((3, 3, 3), dtype=float)
    for ijk, value in beta_df.items():
        i, j, k = ijk_map[ijk[0]], ijk_map[ijk[1]], ijk_map[ijk[2]]
        try:
            beta_np[i, j, k] = float(value)
        except Exception:
            beta_np[i, j, k] = 0.0
    return beta_np


def tensor_from_long(df, mol: str, basis: str, omega: float | int) -> np.ndarray:
    sub = df[
        (df["molecule"] == mol)
        & (df["basis"] == basis)
        & (df["omega"] == float(omega))
    ]
    if sub.shape[0] == 0:
        return np.zeros((3, 3, 3), dtype=float)
    series = {str(ijk): beta for ijk, beta in zip(sub["ijk"], sub["Beta"], strict=False)}
    return _beta_df_to_np(series)


__all__ = [
    "ErrorSettings",
    "compute_error_fields",
    "evaluate_field",
    "load_lebedev_grid",
    "tensor_from_long",
]
