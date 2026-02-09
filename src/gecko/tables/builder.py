from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any

import numpy as np
import pandas as pd

from gecko.core.model import Calculation
from gecko.ids import geom_id_from_molecule, mol_id_from_molecule
from gecko.tables.extractors import (
    extract_alpha,
    extract_beta,
    extract_dipole,
    extract_energy,
    extract_raman,
)


@dataclass
class TableBuilder:
    calcs: list[Calculation]

    def build_geometries(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        seen: set[str] = set()

        for calc in self.calcs:
            qmol = calc.molecule
            if qmol is None:
                continue

            gid = calc.meta.get("geom_id") or geom_id_from_molecule(qmol)
            if gid is None:
                continue
            if gid in seen:
                continue
            seen.add(gid)

            geometry = np.asarray(qmol.geometry, dtype=float).reshape(-1, 3)
            rows.append(
                {
                    "geom_id": gid,
                    "mol_id": calc.meta.get("mol_id") or mol_id_from_molecule(qmol),
                    "label": calc.meta.get("label"),
                    "symbols": list(qmol.symbols),
                    "geometry_angstrom": geometry.tolist(),
                    "charge": getattr(qmol, "charge", None),
                    "multiplicity": getattr(qmol, "multiplicity", None),
                    "source_calc_id": calc.meta.get("calc_id"),
                    "source_root": str(calc.root),
                }
            )

        return pd.DataFrame(rows)

    def build_beta(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_beta(calc))
        return pd.DataFrame(rows)

    def build_alpha(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_alpha(calc))
        return pd.DataFrame(rows)

    def build_energy(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_energy(calc))
        return pd.DataFrame(rows)

    def build_dipole(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_dipole(calc))
        return pd.DataFrame(rows)

    def build_raman(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for calc in self.calcs:
            rows.extend(extract_raman(calc))
        return pd.DataFrame(rows)

    def compare_energy(
        self,
        *,
        ref_basis: str,
        keys: list[str] | None = None,
        rel_eps: float = 1e-12,
        small_thresh: float = 1e-6,
    ) -> pd.DataFrame:
        df = self.build_energy()
        if df.empty:
            return df

        if keys is None:
            keys = ["mol_id", "method"]

        pivot = df.pivot_table(index=keys, columns="basis", values="energy", aggfunc="first")
        pivot = pivot.reset_index()

        if ref_basis not in pivot.columns:
            basis_cols = [c for c in pivot.columns if c not in keys]
            mra_cols = [c for c in basis_cols if str(c).startswith("mra-")]
            ref_basis = (mra_cols[0] if mra_cols else (basis_cols[0] if basis_cols else ref_basis))
            if ref_basis not in pivot.columns:
                return pivot

        basis_cols = [c for c in pivot.columns if c not in keys]
        ref_vals = pivot[ref_basis].astype(float)

        for basis in basis_cols:
            if basis == ref_basis:
                continue
            col = pivot[basis].astype(float)
            delta = col - ref_vals
            rel = np.where(np.abs(ref_vals) > small_thresh, delta / np.abs(ref_vals), np.nan)
            rel = np.where(np.abs(ref_vals) > rel_eps, rel, np.nan)
            pivot[f"delta_{basis}"] = delta
            pivot[f"rel_{basis}"] = rel

        pivot.attrs["ref_basis"] = ref_basis
        return pivot

    def compare_energy_long(
        self,
        *,
        ref_basis: str,
        keys: list[str] | None = None,
        rel_eps: float = 1e-12,
        small_thresh: float = 1e-6,
    ) -> pd.DataFrame:
        df = self.build_energy()
        if df.empty:
            return df

        if keys is None:
            keys = ["mol_id", "method"]

        pivot = df.pivot_table(index=keys, columns="basis", values="energy", aggfunc="first")
        pivot = pivot.reset_index()

        if ref_basis not in pivot.columns:
            basis_cols = [c for c in pivot.columns if c not in keys]
            mra_cols = [c for c in basis_cols if str(c).startswith("mra-")]
            ref_basis = (mra_cols[0] if mra_cols else (basis_cols[0] if basis_cols else ref_basis))
            if ref_basis not in pivot.columns:
                return pd.DataFrame()

        basis_cols = [c for c in pivot.columns if c not in keys]
        ref_vals = pivot[ref_basis].astype(float)

        long_rows: list[dict[str, Any]] = []
        for _, row in pivot.iterrows():
            for basis in basis_cols:
                if basis == ref_basis:
                    continue
                val = float(row[basis])
                ref_val = float(row[ref_basis])
                delta = val - ref_val
                if abs(ref_val) < small_thresh:
                    rel = np.nan
                else:
                    rel = delta / abs(ref_val) if abs(ref_val) > rel_eps else np.nan
                long_rows.append(
                    {
                        **{k: row[k] for k in keys},
                        "ref_basis": ref_basis,
                        "basis": basis,
                        "energy": val,
                        "ref_energy": ref_val,
                        "delta": delta,
                        "rel": rel,
                    }
                )

        return pd.DataFrame(long_rows)

    def compare_alpha_long(
        self,
        *,
        ref_basis: str,
        keys: list[str] | None = None,
        rel_eps: float = 1e-12,
        small_thresh: float = 1e-6,
    ) -> pd.DataFrame:
        df = self.build_alpha()
        if df.empty:
            return df

        if keys is None:
            keys = ["mol_id", "method", "omega", "ij"]

        pivot = df.pivot_table(index=keys, columns="basis", values="value", aggfunc="first")
        pivot = pivot.reset_index()

        if ref_basis not in pivot.columns:
            basis_cols = [c for c in pivot.columns if c not in keys]
            mra_cols = [c for c in basis_cols if str(c).startswith("mra-")]
            ref_basis = (mra_cols[0] if mra_cols else (basis_cols[0] if basis_cols else ref_basis))
            if ref_basis not in pivot.columns:
                return pd.DataFrame()

        basis_cols = [c for c in pivot.columns if c not in keys]

        long_rows: list[dict[str, Any]] = []
        for _, row in pivot.iterrows():
            ref_val = float(row[ref_basis])
            if not np.isfinite(ref_val):
                continue
            for basis in basis_cols:
                if basis == ref_basis:
                    continue
                val = float(row[basis])
                if not np.isfinite(val):
                    continue
                delta = val - ref_val
                if abs(ref_val) < small_thresh:
                    rel = np.nan
                else:
                    rel = delta / abs(ref_val) if abs(ref_val) > rel_eps else np.nan
                long_rows.append(
                    {
                        **{k: row[k] for k in keys},
                        "ref_basis": ref_basis,
                        "basis": basis,
                        "value": val,
                        "ref_value": ref_val,
                        "delta": delta,
                        "rel": rel,
                    }
                )

        return pd.DataFrame(long_rows)

    def compare_beta_long(
        self,
        *,
        ref_basis: str,
        keys: list[str] | None = None,
        rel_eps: float = 1e-12,
        small_thresh: float = 1e-6,
    ) -> pd.DataFrame:
        df = self.build_beta()
        if df.empty:
            return df

        if keys is None:
            keys = ["mol_id", "method", "omegaA", "omegaB", "omegaC", "ijk"]

        pivot = df.pivot_table(index=keys, columns="basis", values="value", aggfunc="first")
        pivot = pivot.reset_index()

        if ref_basis not in pivot.columns:
            basis_cols = [c for c in pivot.columns if c not in keys]
            mra_cols = [c for c in basis_cols if str(c).startswith("mra-")]
            ref_basis = (mra_cols[0] if mra_cols else (basis_cols[0] if basis_cols else ref_basis))
            if ref_basis not in pivot.columns:
                return pd.DataFrame()

        basis_cols = [c for c in pivot.columns if c not in keys]

        long_rows: list[dict[str, Any]] = []
        for _, row in pivot.iterrows():
            ref_val = float(row[ref_basis])
            for basis in basis_cols:
                if basis == ref_basis:
                    continue
                val = float(row[basis])
                delta = val - ref_val
                if abs(ref_val) < small_thresh:
                    rel = np.nan
                else:
                    rel = delta / abs(ref_val) if abs(ref_val) > rel_eps else np.nan
                long_rows.append(
                    {
                        **{k: row[k] for k in keys},
                        "ref_basis": ref_basis,
                        "basis": basis,
                        "value": val,
                        "ref_value": ref_val,
                        "delta": delta,
                        "rel": rel,
                    }
                )

        return pd.DataFrame(long_rows)

    def compare_raman_long(
        self,
        *,
        ref_basis: str,
        property_name: str,
        keys: list[str] | None = None,
        rel_eps: float = 1e-12,
        small_thresh: float = 1e-6,
    ) -> pd.DataFrame:
        df = self.build_raman()
        if df.empty:
            return df

        if property_name not in df.columns:
            return pd.DataFrame()

        if keys is None:
            keys = ["mol_id", "method", "omega_pol", "mode"]

        pivot = df.pivot_table(index=keys, columns="basis", values=property_name, aggfunc="first")
        pivot = pivot.reset_index()

        freq_pivot = df.pivot_table(index=keys, columns="basis", values="freq_cm1", aggfunc="first")
        freq_pivot = freq_pivot.reset_index()

        if ref_basis not in pivot.columns:
            basis_cols = [c for c in pivot.columns if c not in keys]
            mra_cols = [c for c in basis_cols if str(c).startswith("mra-")]
            ref_basis = (mra_cols[0] if mra_cols else (basis_cols[0] if basis_cols else ref_basis))
            if ref_basis not in pivot.columns:
                return pd.DataFrame()

        basis_cols = [c for c in pivot.columns if c not in keys]

        long_rows: list[dict[str, Any]] = []
        freq_by_key = {
            tuple(freq_row[k] for k in keys): freq_row
            for _, freq_row in freq_pivot.iterrows()
        }

        for _, row in pivot.iterrows():
            ref_val = float(row[ref_basis])
            freq_row = freq_by_key.get(tuple(row[k] for k in keys), {})
            ref_freq = float(freq_row.get(ref_basis)) if ref_basis in freq_row else np.nan
            for basis in basis_cols:
                if basis == ref_basis:
                    continue
                val = float(row[basis])
                freq_val = float(freq_row.get(basis)) if basis in freq_row else np.nan
                delta = val - ref_val
                if abs(ref_val) < small_thresh:
                    rel = np.nan
                else:
                    rel = delta / abs(ref_val) if abs(ref_val) > rel_eps else np.nan
                long_rows.append(
                    {
                        **{k: row[k] for k in keys},
                        "ref_basis": ref_basis,
                        "basis": basis,
                        "property": property_name,
                        "value": val,
                        "ref_value": ref_val,
                        "freq_cm1": freq_val,
                        "ref_freq_cm1": ref_freq,
                        "delta": delta,
                        "rel": rel,
                    }
                )

        return pd.DataFrame(long_rows)
