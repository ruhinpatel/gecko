# src/gecko/plugins/madness/parse.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable
import numpy as np

from gecko.core.model import Calculation



_IJK = ["x", "y", "z"]

def _beta_df_to_tensor(beta_df) -> dict[str, Any]:
    """
    Convert MADQC beta_pivot (a pandas DataFrame pivot_table) into a tensor-first
    representation.

    Expects:
      beta_df.index = MultiIndex [omegaA, omegaB, omegaC]
      beta_df.columns = strings like "xyz", "xxy", etc.
      beta_df.values = scalar (float or complex)

    Returns:
      {
        "omega": np.ndarray shape (n_freq, 3) [omegaA, omegaB, omegaC],
        "components": list[str] length n_comp,
        "values": np.ndarray shape (n_freq, n_comp),
        "shape": ("freq", "component"),
      }
    """
    # Lazy import: parse.py can be imported without pandas installed
    import pandas as pd  # type: ignore

    if beta_df is None:
        return {}

    if not isinstance(beta_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(beta_df)}")

    # Ensure stable ordering for reproducibility
    beta_df = beta_df.sort_index()
    beta_df = beta_df.reindex(sorted(beta_df.columns), axis=1)

    # Extract omega tuples from MultiIndex
    omega = np.asarray(beta_df.index.to_list(), dtype=float)  # (n_freq, 3)

    components = [str(c) for c in beta_df.columns.to_list()]
    values = beta_df.to_numpy()  # (n_freq, n_comp)

    return {
        "omega": omega,
        "components": components,
        "values": values,
        "shape": ("freq", "component"),
    }



def parse_run(calc: Calculation) -> None:
    ci_path = calc.artifacts.get("calc_info_json")
    if not ci_path:
        return

    calc.meta["style"] = "madqc"

    from gecko.plugins.madness.legacy.madness_data import madqc_parser
    obj = madqc_parser(ci_path)

    # Keep raw JSON as a source of truth during migration
    calc.data["calc_info"] = _read_json(ci_path)

    # Tensor-first hyperpolarizability
    calc.data["beta"] = _beta_df_to_tensor(obj.beta_pivot)

    # (Optional) do the same for alpha later:
    # calc.data["alpha"] = _alpha_df_to_tensor(obj.alpha_pivot)

    # Keep other useful arrays
    calc.data["orbital_energies"] = obj.orbital_energies
    calc.data["hessian"] = obj.hessian
    calc.data["normal_modes"] = obj.normal_modes
    calc.data["vibrational_frequencies"] = obj.vibrational_frequencies
    calc.data["polarization_frequencies"] = getattr(obj, "polarization_frequencies", None)

    # Optional
    calc.data["molecule"] = obj.molecule
    calc.meta["ground_state_energy"] = obj.ground_state_energy


def _parse_madqc_calc_info(path: Path) -> dict[str, Any]:
    # Import inside function so gecko can still import even if optional deps missing
    from gecko.plugins.madness.legacy.madness_data import madqc_parser

    obj = madqc_parser(path)

    # NOTE: obj currently stores a bunch of pandas DataFrames and numpy arrays.
    # That's fine for migration. We'll normalize later.
    return {
        "raw": _read_json(path),  # keep the raw JSON dict as source of truth
        "ground_state_energy": obj.ground_state_energy,
        "molecule": obj.molecule,
        "orbital_energies": obj.orbital_energies,
        "alpha_df": obj.alpha_pivot,
        "beta_df": obj.beta_pivot,
        "raman_df": obj.raman_pivot,
        "hessian": obj.hessian,
        "normal_modes": obj.normal_modes,
        "polarization_frequencies": getattr(obj, "polarization_frequencies", None),
        "vibrational_frequencies": obj.vibrational_frequencies,
    }


def _read_json(path: Path) -> dict[str, Any]:
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
