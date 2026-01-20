from pathlib import Path
from typing import Iterable

import pandas as pd

from gecko.index import CalcIndex
from gecko.tables import TableBuilder


def build_beta_table(
    calc_dirs: Iterable[str | Path],
    *,
    verbose: bool = True,
    fail_fast: bool = False,
    require_geometry: bool = False,
    inline_geometry: bool = False,
    include_geometry: bool = False,
    mol_file: str | Path | None = None,
    mol_dir: str | Path | None = None,
    mol_root: str | Path | None = None,
    mol_map: str | Path | None = None,
) -> pd.DataFrame:
    """
    Build a long-form table for hyperpolarizability data.

    If a calculation fails to load or parse, the offending directory
    is reported and skipped (unless fail_fast=True).

        Output columns:
            - calc_id
            - geom_id
            - mol_id
            - molecule_id
            - label
            - code
            - root
            - basis
            - method
            - omegaA, omegaB, omegaC
            - ijk
            - value
            - geometry (optional JSON string)
    """
    if inline_geometry and not include_geometry:
        include_geometry = True

    index = CalcIndex.from_dirs(
        calc_dirs,
        mol_root=mol_root,
        mol_map=mol_map,
        mol_file=mol_file,
        mol_dir=mol_dir,
        strict=fail_fast,
    )

    if verbose:
        for failure in index.failures:
            print("\n[gecko] FAILED to process calculation:")
            print(f"  path: {failure.path}")
            print(f"  error: {failure.error}")

    if require_geometry:
        filtered = []
        for calc in index.calcs:
            if calc.meta.get("geom_id") is None:
                msg = f"[gecko] missing geometry, skipping: {calc.root}"
                if verbose:
                    print(msg)
                if fail_fast:
                    raise ValueError(msg)
                continue
            filtered.append(calc)
        index.calcs = filtered

    builder = TableBuilder(index.calcs)
    beta_df = builder.build_beta()

    if include_geometry:
        import json

        geom_df = builder.build_geometries()
        if not geom_df.empty:
            beta_df = beta_df.merge(
                geom_df[["geom_id", "symbols", "geometry_angstrom"]],
                on="geom_id",
                how="left",
            )
            if not beta_df.empty:
                def _format_geometry(row):
                    symbols = row.get("symbols")
                    geometry = row.get("geometry_angstrom")
                    if symbols is None or geometry is None:
                        return None
                    if isinstance(symbols, float) and pd.isna(symbols):
                        return None
                    if isinstance(geometry, float) and pd.isna(geometry):
                        return None
                    return json.dumps(
                        {"symbols": symbols, "geometry": geometry},
                        separators=(",", ":"),
                    )

                beta_df["geometry"] = beta_df.apply(_format_geometry, axis=1)
                beta_df = beta_df.drop(columns=["symbols", "geometry_angstrom"])

    return beta_df
