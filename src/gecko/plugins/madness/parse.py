from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gecko.core.model import Calculation


def _beta_df_to_tensor(beta_df) -> dict[str, Any]:
    """
    Convert MADNESS beta_pivot (pandas DataFrame) into a tensor-first representation.

    Expects:
      - beta_df.index = MultiIndex [omegaA, omegaB, omegaC]
      - beta_df.columns = strings like "xyz", "xxy", etc.
      - beta_df.values = scalar (float or complex)

    Returns:
      {
        "omega": np.ndarray shape (n_freq, 3) [omegaA, omegaB, omegaC],
        "components": list[str],
        "values": np.ndarray shape (n_freq, n_comp),
        "shape": ("freq", "component"),
      }
    """
    import pandas as pd  # lazy import

    if beta_df is None:
        return {}

    if not isinstance(beta_df, pd.DataFrame):
        raise TypeError(f"Expected pandas DataFrame, got {type(beta_df)}")

    beta_df = beta_df.sort_index()
    beta_df = beta_df.reindex(sorted(beta_df.columns), axis=1)

    omega = np.asarray(beta_df.index.to_list(), dtype=float)  # (n_freq, 3)
    components = [str(c) for c in beta_df.columns.to_list()]
    values = beta_df.to_numpy()  # (n_freq, n_comp)

    return {"omega": omega, "components": components, "values": values, "shape": ("freq", "component")}


def parse_run(calc: Calculation) -> None:
    """
    Populate calc.data for MADNESS runs.

    Supports:
      - MADQC style: *.calc_info.json
      - Legacy molresponse style: output.json

    Both are parsed by the same legacy madqc_parser (schema is nearly the same),
    with component label normalization handled inside the legacy parser.
    """
    json_path, style = _select_input_json(calc)
    if json_path is None:
        return

    calc.meta["style"] = style

    from gecko.plugins.madness.legacy.madness_data import madqc_parser

    obj = madqc_parser(json_path)

    # Keep raw json as source of truth during migration
    calc.data["raw_json"] = _read_json(json_path)

    # Tensor-first hyperpolarizability
    calc.data["beta"] = _beta_df_to_tensor(obj.beta_pivot)

    # Keep other useful arrays (best-effort; may be None for some runs)
    calc.data["orbital_energies"] = obj.orbital_energies
    calc.data["hessian"] = obj.hessian
    calc.data["normal_modes"] = obj.normal_modes
    calc.data["vibrational_frequencies"] = obj.vibrational_frequencies
    calc.data["polarization_frequencies"] = getattr(obj, "polarization_frequencies", None)

    calc.data["molecule"] = obj.molecule
    calc.meta["ground_state_energy"] = obj.ground_state_energy


def _select_input_json(calc: Calculation) -> tuple[Path | None, str | None]:
    """
    Decide which JSON file to parse for MADNESS.

    Priority:
      1) MADQC: calc_info_json
      2) legacy: output_json
    """
    ci_path = calc.artifacts.get("calc_info_json")
    if ci_path and ci_path.exists():
        return ci_path, "madqc"

    out_path = calc.artifacts.get("output_json")
    if out_path and out_path.exists():
        return out_path, "molresponse"

    return None, None


def _read_json(path: Path) -> dict[str, Any]:
    import json
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)
