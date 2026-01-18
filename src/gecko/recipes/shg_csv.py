from pathlib import Path
from typing import Iterable

import pandas as pd
import gecko


def build_beta_table(
    calc_dirs: Iterable[str | Path],
    *,
    verbose: bool = True,
    fail_fast: bool = False,
) -> pd.DataFrame:
    """
    Build a long-form table for hyperpolarizability data.

    If a calculation fails to load or parse, the offending directory
    is reported and skipped (unless fail_fast=True).

    Output columns:
      - root
      - molecule (if available)
      - basis (if available)
      - omegaA, omegaB, omegaC
      - ijk
      - value
    """
    rows: list[dict] = []

    for d in calc_dirs:
        try:
            calc = gecko.load_calc(d)

            beta = calc.data.get("beta") or {}
            if not beta:
                if verbose:
                    print(f"[gecko] no beta data found, skipping: {d}")
                continue

            omega = beta["omega"]          # shape (N, 3)
            comps = beta["components"]     # len n_comp
            vals = beta["values"]          # shape (N, n_comp)

            mol = (
                calc.meta.get("molecule")
                or calc.data.get("calc_info", {}).get("molecule")
                or calc.data.get("raw_json", {}).get("molecule")
            )
            basis = calc.meta.get("basis")

            for i in range(vals.shape[0]):
                omegaA, omegaB, omegaC = omega[i, :]
                for j, ijk in enumerate(comps):
                    rows.append(
                        {
                            "root": str(calc.root),
                            "molecule": mol,
                            "basis": basis,
                            "omegaA": float(omegaA),
                            "omegaB": float(omegaB),
                            "omegaC": float(omegaC),
                            "ijk": ijk,
                            "value": vals[i, j],
                        }
                    )

        except Exception as e:
            if verbose:
                print("\n[gecko] FAILED to process calculation:")
                print(f"  path: {d}")
                print(f"  error: {type(e).__name__}: {e}")
            if fail_fast:
                raise

    return pd.DataFrame(rows)
