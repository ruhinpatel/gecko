from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from gecko.core.iterators import iter_calc_dirs
from gecko.recipes.shg_csv import build_beta_table


def _normalize_long_df(df: pd.DataFrame) -> pd.DataFrame:
    for col in ("molecule", "basis", "ijk"):
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", "", regex=True)
            )
    if "omega" in df.columns:
        df["omega"] = df["omega"].astype(float)
    return df


def load_shg_df_from_csv(path: Path) -> pd.DataFrame:
    csv_path = Path(path).expanduser().resolve()
    df = pd.read_csv(csv_path)
    return _normalize_long_df(df)


def build_shg_df_from_db(
    db_dir: Path,
    *,
    start_at: int = 0,
    tol: float = 1e-12,
    include_geometry: bool = True,
    verbose: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    root = Path(db_dir).expanduser().resolve()
    calc_dirs = list(iter_calc_dirs(root))
    df = build_beta_table(
        calc_dirs,
        shg_only=True,
        add_shg_omega=True,
        shg_start_at=start_at,
        shg_tol=tol,
        include_geometry=include_geometry,
        app_compat=True,
        verbose=verbose,
        fail_fast=fail_fast,
    )
    return _normalize_long_df(df)
