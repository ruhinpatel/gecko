from __future__ import annotations

from pathlib import Path
from typing import Iterable, Any

import pandas as pd
import numpy as np
import json

import gecko
from gecko.core import iterators
from gecko.ids import geom_id_from_molecule, mol_id_from_molecule
from gecko.plugins.dalton.detect import can_load as dalton_can_load
from gecko.plugins.madness.detect import can_load as madness_can_load


def _expand_calc_paths(calc_dirs: Iterable[str | Path] | str | Path) -> list[Path]:
    if isinstance(calc_dirs, (str, Path)):
        roots: list[str | Path] = [calc_dirs]
    else:
        roots = list(calc_dirs)

    paths: list[Path] = []
    for entry in roots:
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            if madness_can_load(path) or dalton_can_load(path):
                paths.append(path)
            paths.extend(list(iterators.iter_calc_dirs(path)))
        else:
            paths.append(path)

    seen: set[Path] = set()
    ordered: list[Path] = []
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        ordered.append(path)
    return ordered


def _build_beta_rows(
    calc,
    *,
    include_geometry: bool,
    require_geometry: bool,
) -> list[dict[str, Any]]:
    beta = calc.data.get("beta") or {}
    if not beta or not all(k in beta for k in ("omega", "components", "values")):
        return []

    qmol = calc.molecule
    if require_geometry and qmol is None:
        return []

    gid = geom_id_from_molecule(qmol)
    mid = mol_id_from_molecule(qmol)

    omega = np.asarray(beta["omega"], dtype=float)
    comps = list(beta["components"])
    vals = np.asarray(beta["values"], dtype=float)

    geometry_json = None
    if include_geometry and qmol is not None:
        geometry = np.asarray(qmol.geometry, dtype=float).reshape(-1, 3)
        geometry_json = {
            "symbols": list(qmol.symbols),
            "geometry": geometry.tolist(),
        }

    rows: list[dict[str, Any]] = []
    for i in range(vals.shape[0]):
        omegaA, omegaB, omegaC = map(float, omega[i])
        for j, ijk in enumerate(comps):
            row = {
                "geom_id": gid,
                "mol_id": mid,
                "molecule_id": gid,
                "code": calc.code,
                "root": str(calc.root),
                "basis": calc.meta.get("basis"),
                "method": calc.meta.get("method"),
                "omegaA": omegaA,
                "omegaB": omegaB,
                "omegaC": omegaC,
                "ijk": str(ijk).lower(),
                "value": float(vals[i, j]),
            }
            if include_geometry:
                row["geometry"] = (
                    None
                    if geometry_json is None
                    else json.dumps(geometry_json, separators=(",", ":"))
                )
            rows.append(row)

    return rows


def _assign_shg_omega(
    df: pd.DataFrame,
    *,
    tol: float,
    start_at: int,
    shg_only: bool,
) -> pd.DataFrame:
    if df.empty:
        return df

    mask = (df["omegaB"] - df["omegaC"]).abs() <= tol

    if "omega" not in df.columns:
        df["omega"] = pd.Series([pd.NA] * len(df), index=df.index)

    shg_df = df[mask]
    if not shg_df.empty:
        def _group_key(row: pd.Series) -> str:
            gid = row.get("geom_id")
            if isinstance(gid, str) and gid:
                return f"geom:{gid}"
            mid = row.get("mol_id")
            if isinstance(mid, str) and mid:
                return f"mol:{mid}"
            root = row.get("root")
            return f"root:{root}"

        keys = shg_df.apply(_group_key, axis=1)
        for key in keys.unique():
            group = shg_df[keys == key]
            unique_freqs = sorted(group["omegaB"].dropna().unique())
            mapping = {float(freq): idx + start_at for idx, freq in enumerate(unique_freqs)}
            df.loc[group.index, "omega"] = group["omegaB"].map(mapping)

    if shg_only:
        return df[mask].copy()

    return df


def build_beta_table(
    calc_dirs: Iterable[str | Path] | str | Path,
    *,
    verbose: bool = True,
    fail_fast: bool = False,
    require_geometry: bool = False,
    inline_geometry: bool = False,
    include_geometry: bool = False,
    shg_only: bool = False,
    shg_tol: float = 1e-12,
    shg_start_at: int = 0,
    add_shg_omega: bool = False,
    app_compat: bool = False,
) -> pd.DataFrame:
    """
    Build a long-form table for hyperpolarizability data.

    If a calculation fails to load or parse, the offending directory
    is reported and skipped (unless fail_fast=True).

        Output columns:
            - geom_id
            - mol_id
            - molecule_id
            - code
            - root
            - basis
            - method
            - omegaA, omegaB, omegaC
            - ijk
            - value
            - geometry (optional JSON string)

        SHG options:
            - shg_only: keep only rows where omegaB == omegaC (within tol)
            - add_shg_omega: assign per-molecule omega index starting at 0
            - shg_only implies include_geometry=True unless explicitly overridden
            - app_compat: add "molecule" and "Beta" columns for legacy viz apps
    """
    if inline_geometry and not include_geometry:
        include_geometry = True

    if shg_only and not include_geometry:
        include_geometry = True

    if shg_only and not add_shg_omega:
        add_shg_omega = True

    calc_paths = _expand_calc_paths(calc_dirs)

    failures: list[tuple[str, str]] = []
    rows: list[dict[str, Any]] = []

    for path in calc_paths:
        try:
            calcs = gecko.load_calcs(path)
        except Exception as exc:
            failures.append((str(path), f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                raise
            continue

        for calc in calcs:
            if require_geometry and calc.molecule is None:
                msg = f"[gecko] missing geometry, skipping: {calc.root}"
                if verbose:
                    print(msg)
                if fail_fast:
                    raise ValueError(msg)
                continue

            rows.extend(
                _build_beta_rows(
                    calc,
                    include_geometry=include_geometry,
                    require_geometry=require_geometry,
                )
            )

    if verbose:
        for path, error in failures:
            print("\n[gecko] FAILED to process calculation:")
            print(f"  path: {path}")
            print(f"  error: {error}")

    beta_df = pd.DataFrame(rows)

    if add_shg_omega or shg_only:
        beta_df = _assign_shg_omega(
            beta_df,
            tol=shg_tol,
            start_at=shg_start_at,
            shg_only=shg_only,
        )

    if app_compat and not beta_df.empty:
        beta_df["molecule"] = beta_df["mol_id"]
        beta_df["Beta"] = beta_df["value"]

    return beta_df
