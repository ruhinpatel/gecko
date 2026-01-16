from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import gecko


def build_beta_table(calc_dirs: Iterable[str | Path]) -> pd.DataFrame:
    """
    Build a long-form table for hyperpolarizability data.

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
        calc = gecko.load_calc(d)

        beta = calc.data.get("beta") or {}
        if not beta:
            continue

        omega = beta["omega"]          # shape (N, 3)
        comps = beta["components"]     # len 27
        vals = beta["values"]          # shape (N, 27)

        # convenience metadata (best-effort)
        mol = calc.meta.get("molecule") or calc.data.get("calc_info", {}).get("molecule")
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

    return pd.DataFrame(rows)
