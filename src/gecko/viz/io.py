from __future__ import annotations

from pathlib import Path
from typing import Iterable, Dict, Any
import json

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
    if "ijk" in df.columns:
        df["ijk"] = df["ijk"].astype(str).str.upper()
    if "omega" in df.columns:
        df["omega"] = df["omega"].astype(float)
    return df


def geometry_map_from_df(df: pd.DataFrame, *, key: str = "molecule") -> Dict[str, Dict[str, Any]]:
    """Extract a unique geometry map from a long-form SHG dataframe.

    Returns a mapping of label -> {"symbols": [...], "geometry": [[x,y,z], ...]}.
    """
    if df.empty or "geometry" not in df.columns:
        return {}

    if key not in df.columns:
        for candidate in ("molecule", "mol_id", "geom_id", "molecule_id"):
            if candidate in df.columns:
                key = candidate
                break
        else:
            return {}

    geom_map: Dict[str, Dict[str, Any]] = {}
    subset = df[[key, "geometry"]].dropna().drop_duplicates()
    for label, geom in subset.itertuples(index=False):
        label_str = str(label).strip()
        if not label_str or label_str in geom_map:
            continue

        payload = geom
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception:
                continue
        if not isinstance(payload, dict):
            continue
        if "symbols" not in payload or "geometry" not in payload:
            continue
        geom_map[label_str] = payload

    return geom_map


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


def write_beta_viewer_bundle(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    csv_name: str = "shg_ijk.csv",
    geometry_name: str = "geometries.json",
) -> Path:
    """Write a beta-viewer bundle directory with CSV + geometry map."""
    out_path = Path(out_dir).expanduser().resolve()
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / csv_name
    df.to_csv(csv_path, index=False)

    geom_map = geometry_map_from_df(df)
    if geom_map:
        geom_path = out_path / geometry_name
        geom_path.write_text(json.dumps(geom_map, indent=2), encoding="utf-8")

    return csv_path
