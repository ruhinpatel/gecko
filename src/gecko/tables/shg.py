from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Any

import numpy as np
import pandas as pd

import gecko
from gecko.ids import geom_id_from_molecule, mol_id_from_molecule


@dataclass
class _LoadFailure:
    path: str
    error: str


def _load_calcs(path: Path):
    if path.is_dir():
        return gecko.load_calcs(path)
    return [gecko.load_calc(path)]


def build_beta_long(
    calc_paths: Iterable[str | Path],
    *,
    verbose: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    failures: list[_LoadFailure] = []

    for p in calc_paths:
        path = Path(p).expanduser().resolve()
        try:
            calcs = _load_calcs(path)
        except Exception as exc:
            failures.append(_LoadFailure(path=str(path), error=f"{type(exc).__name__}: {exc}"))
            if fail_fast:
                raise
            continue

        for calc in calcs:
            beta = calc.data.get("beta") or {}
            if not beta:
                continue
            if not all(k in beta for k in ("omega", "components", "values")):
                continue

            omega = np.asarray(beta["omega"], dtype=float)
            comps = list(beta["components"])
            vals = np.asarray(beta["values"], dtype=float)

            qmol = calc.molecule
            gid = geom_id_from_molecule(qmol)
            mid = mol_id_from_molecule(qmol)

            for i in range(vals.shape[0]):
                omegaA, omegaB, omegaC = map(float, omega[i])
                for j, ijk in enumerate(comps):
                    rows.append(
                        {
                            "geom_id": gid,
                            "mol_id": mid,
                            "code": calc.code,
                            "basis": calc.meta.get("basis"),
                            "omegaA": omegaA,
                            "omegaB": omegaB,
                            "omegaC": omegaC,
                            "ijk": str(ijk).lower(),
                            "value": float(vals[i, j]),
                            "root": str(calc.root),
                        }
                    )

    if verbose:
        for failure in failures:
            print("\n[gecko] FAILED to process calculation:")
            print(f"  path: {failure.path}")
            print(f"  error: {failure.error}")

    return pd.DataFrame(rows)


def filter_shg_rows(beta_long: pd.DataFrame, *, tol: float = 1e-12) -> pd.DataFrame:
    if beta_long.empty:
        return beta_long.copy()
    df = beta_long.copy()
    mask = (df["omegaB"] - df["omegaC"]).abs() <= tol
    return df[mask].copy()


def _group_key(row: pd.Series) -> str:
    gid = row.get("geom_id")
    if isinstance(gid, str) and gid:
        return f"geom:{gid}"
    mid = row.get("mol_id")
    if isinstance(mid, str) and mid:
        return f"mol:{mid}"
    root = row.get("root")
    return f"root:{root}"


def assign_shg_omega_index(shg_df: pd.DataFrame, *, start_at: int = 0) -> pd.DataFrame:
    if shg_df.empty:
        out = shg_df.copy()
        out["omega"] = pd.Series(dtype=int)
        return out

    df = shg_df.copy()
    df["_group_key"] = df.apply(_group_key, axis=1)

    omega_values = pd.Series(index=df.index, dtype=int)
    for _, group in df.groupby("_group_key", sort=False):
        freqs = group["omegaB"].dropna().unique()
        freqs_sorted = np.sort(freqs)
        mapping = {float(freq): int(idx + start_at) for idx, freq in enumerate(freqs_sorted)}
        omega_values.loc[group.index] = group["omegaB"].map(mapping).astype(int)

    df["omega"] = omega_values
    df = df.drop(columns=["_group_key"])
    return df


def build_shg_ijk(
    calc_paths: Iterable[str | Path],
    *,
    start_at: int = 0,
    tol: float = 1e-12,
    verbose: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    beta_long = build_beta_long(calc_paths, verbose=verbose, fail_fast=fail_fast)
    shg_only = filter_shg_rows(beta_long, tol=tol)
    return assign_shg_omega_index(shg_only, start_at=start_at)
