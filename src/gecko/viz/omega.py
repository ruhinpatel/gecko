from __future__ import annotations

import pandas as pd
import numpy as np


def assign_shg_omega_index(
    df: pd.DataFrame,
    *,
    mol_col: str = "molecule",
    b_col: str = "omegaB",
    c_col: str = "omegaC",
    out_col: str = "omega",
    tol: float = 1e-12,
    start_at: int = 0,
) -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out[out_col] = pd.Series(dtype=float)
        return out

    out = df.copy()
    mask = (out[b_col] - out[c_col]).abs() <= tol

    if out_col not in out.columns:
        out[out_col] = pd.Series([pd.NA] * len(out), index=out.index)

    shg = out[mask]
    if shg.empty:
        return out

    for mol, group in shg.groupby(mol_col, sort=False):
        unique_freqs = sorted(group[b_col].dropna().unique())
        mapping = {float(freq): idx + start_at for idx, freq in enumerate(unique_freqs)}
        out.loc[group.index, out_col] = group[b_col].map(mapping)

    return out
