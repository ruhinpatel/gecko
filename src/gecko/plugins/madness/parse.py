from __future__ import annotations


from pathlib import Path
from typing import Any
import logging


import numpy as np

from gecko.core.model import Calculation
from gecko.molecule.canonical import canonicalize_atom_order

logger = logging.getLogger(__name__)


def _beta_df_to_tensor(beta_df) -> dict[str, Any]:
    """
    Convert MADNESS df_pivot (pandas DataFrame) into a tensor-first representation.

    Expects:
      - beta_df.index = MultiIndex [omegaA, omegaB, omegaC]
      - beta_df.columns = strings like "xyz", "xxy", etc.
      - beta_df.values = scalar (float or complex)

    Returns:
      {
        "omega": np.ndarray shape (n_freq, num_compents) -> example for beta [omegaA, omegaB, omegaC],
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
    values = beta_df.to_numpy(dtype=float)  # (n_freq, n_comp)
    # zero out tiny values for cleaner output
    #values[np.abs(values) < 1e-3] = 0.0

    return {"omega": omega, "components": components, "values": values, "shape": ("freq", "component")}


def _tensor_has_rows(tensor: Any) -> bool:
    if not isinstance(tensor, dict):
        return False
    if not all(k in tensor for k in ("omega", "components", "values")):
        return False
    values = np.asarray(tensor.get("values"))
    components = tensor.get("components") or []
    return values.ndim == 2 and values.shape[0] > 0 and values.shape[1] > 0 and len(components) > 0


def _legacy_alpha_to_tensor(raw_json: dict[str, Any]) -> dict[str, Any]:
    response = raw_json.get("response")
    if not isinstance(response, dict):
        return {}
    alpha = response.get("alpha")
    if not isinstance(alpha, dict):
        return {}

    values = alpha.get("alpha")
    components = alpha.get("ij")
    omega = alpha.get("omega")
    if not (isinstance(values, list) and isinstance(components, list) and isinstance(omega, list)):
        return {}
    if not values or len(values) != len(components) or len(values) != len(omega):
        return {}

    by_freq: dict[float, dict[str, float]] = {}
    for freq_raw, comp_raw, value_raw in zip(omega, components, values, strict=False):
        try:
            freq = float(freq_raw)
            comp = str(comp_raw).strip().lower()
            value = float(value_raw)
        except Exception:
            continue
        if not comp:
            continue
        by_freq.setdefault(freq, {})[comp] = value

    if not by_freq:
        return {}

    freqs = sorted(by_freq.keys())
    comps = sorted({comp for comp_map in by_freq.values() for comp in comp_map.keys()})
    arr = np.full((len(freqs), len(comps)), np.nan, dtype=float)
    for i, freq in enumerate(freqs):
        for j, comp in enumerate(comps):
            if comp in by_freq[freq]:
                arr[i, j] = by_freq[freq][comp]

    return {"omega": np.asarray(freqs, dtype=float), "components": comps, "values": arr, "shape": ("freq", "component")}


def _legacy_beta_to_tensor(raw_json: dict[str, Any]) -> dict[str, Any]:
    hyper = raw_json.get("hyper")
    if not isinstance(hyper, dict):
        return {}
    beta = hyper.get("beta")
    if not isinstance(beta, dict):
        return {}

    a = beta.get("A")
    b = beta.get("B")
    c = beta.get("C")
    a_freq = beta.get("Afreq")
    b_freq = beta.get("Bfreq")
    c_freq = beta.get("Cfreq")
    values = beta.get("Beta")

    if not all(isinstance(x, list) for x in (a, b, c, a_freq, b_freq, c_freq, values)):
        return {}
    n = len(values)
    if n == 0:
        return {}
    if not all(len(x) == n for x in (a, b, c, a_freq, b_freq, c_freq)):
        return {}

    by_freq: dict[tuple[float, float, float], dict[str, float]] = {}
    for a_raw, b_raw, c_raw, af_raw, bf_raw, cf_raw, val_raw in zip(
        a, b, c, a_freq, b_freq, c_freq, values, strict=False
    ):
        try:
            comp = f"{str(a_raw).strip()}{str(b_raw).strip()}{str(c_raw).strip()}".lower()
            freq = (float(af_raw), float(bf_raw), float(cf_raw))
            val = float(val_raw)
        except Exception:
            continue
        if len(comp) != 3:
            continue
        by_freq.setdefault(freq, {})[comp] = val

    if not by_freq:
        return {}

    freqs = sorted(by_freq.keys())
    comps = sorted({comp for comp_map in by_freq.values() for comp in comp_map.keys()})
    arr = np.full((len(freqs), len(comps)), np.nan, dtype=float)
    for i, freq in enumerate(freqs):
        for j, comp in enumerate(comps):
            if comp in by_freq[freq]:
                arr[i, j] = by_freq[freq][comp]

    return {
        "omega": np.asarray(freqs, dtype=float),
        "components": comps,
        "values": arr,
        "shape": ("freq", "component"),
    }


def _format_mra_threshold(prefix: str, value: float) -> str:
    try:
        import math

        if value <= 0:
            return "mra"
        exp = int(round(-math.log10(float(value))))
        if exp < 0:
            return "mra"
        return f"mra-{prefix}{exp:02d}"
    except Exception:
        return "mra"


def _infer_mra_basis_from_obj(obj: Any) -> str | None:
    if not isinstance(obj, (dict, list)):
        return None

    # Depth-first search for known keys.
    if isinstance(obj, dict):
        # Direct dconv
        for k in ("dconv", "converged_for_dconv"):
            v = obj.get(k)
            if isinstance(v, (int, float)) and float(v) > 0:
                return _format_mra_threshold("d", float(v))

        # Protocol (list)
        v = obj.get("protocol")
        if isinstance(v, list) and v:
            last = v[-1]
            if isinstance(last, (int, float)) and float(last) > 0:
                return _format_mra_threshold("p", float(last))

        # Some calc-info payloads store convergence thresholds under different keys.
        for k in ("converged_for_thresh", "thresh"):
            v = obj.get(k)
            if isinstance(v, (int, float)) and float(v) > 0:
                return _format_mra_threshold("p", float(v))

        for v in obj.values():
            out = _infer_mra_basis_from_obj(v)
            if out is not None:
                return out
        return None

    # list
    for it in obj:
        out = _infer_mra_basis_from_obj(it)
        if out is not None:
            return out
    return None


def _infer_mra_basis_from_input_in_text(text: str) -> str | None:
    import re

    # dconv 1e-6 (or dconv=1e-6)
    m = re.search(r"(?im)^\s*dconv\s*(?:=)?\s*([0-9.+-eE]+)\s*$", text)
    if m:
        try:
            return _format_mra_threshold("d", float(m.group(1)))
        except Exception:
            pass

    # protocol [0.0001,1e-06,1e-7]
    m = re.search(r"(?im)^\s*protocol\s*(?:=)?\s*\[(.*?)\]\s*$", text)
    if m:
        raw = m.group(1)
        nums = re.findall(r"[0-9.]+(?:[eE][+-]?[0-9]+)?", raw)
        if nums:
            try:
                return _format_mra_threshold("p", float(nums[-1]))
            except Exception:
                pass

    # protocol 1e-4 1e-6 1e-8 (space-separated)
    m = re.search(r"(?im)^\s*protocol\s+(.+?)\s*$", text)
    if m:
        nums = re.findall(r"[0-9.]+(?:[eE][+-]?[0-9]+)?", m.group(1))
        if nums:
            try:
                return _format_mra_threshold("p", float(nums[-1]))
            except Exception:
                pass

    return None


def _infer_method_from_input_in_text(text: str) -> str | None:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
    lower_lines = [ln.lower() for ln in lines]

    for ln in lower_lines:
        if ln.startswith("xc"):
            parts = ln.split()
            if len(parts) >= 2:
                return parts[1].upper()
            return None

    if any(ln.startswith("mp2") for ln in lower_lines):
        return "MP2"
    if any(ln.startswith("hf") for ln in lower_lines):
        return "HF"
    if any(ln.startswith("dft") for ln in lower_lines):
        return "HF"
    return None


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

    # Basis label (MRA) inference.
    # Priority: paired input .in (MADQC) -> input.json -> parsed payload.
    basis = None
    input_in = calc.artifacts.get("input_in")
    if isinstance(input_in, Path) and input_in.exists():
        try:
            basis = _infer_mra_basis_from_input_in_text(
                input_in.read_text(encoding="utf-8", errors="ignore")
            )
        except Exception:
            basis = None
        if calc.meta.get("method") is None:
            try:
                calc.meta["method"] = _infer_method_from_input_in_text(
                    input_in.read_text(encoding="utf-8", errors="ignore")
                )
            except Exception:
                pass

    input_json = calc.artifacts.get("input_json")
    if isinstance(input_json, Path) and input_json.exists():
        try:
            basis = _infer_mra_basis_from_obj(_read_json(input_json))
        except Exception:
            basis = None
    if basis is None:
        basis = _infer_mra_basis_from_obj(calc.data.get("raw_json"))
    calc.basis = basis or "mra"
    calc.meta["basis"] = calc.basis

    # Tensor-first hyperpolarizability
    calc.data["beta"] = _beta_df_to_tensor(obj.beta_pivot)
    if not _tensor_has_rows(calc.data["beta"]):
        calc.data["beta"] = _legacy_beta_to_tensor(calc.data["raw_json"])

    calc.data["alpha"] = _beta_df_to_tensor(obj.alpha_pivot)
    if not _tensor_has_rows(calc.data["alpha"]):
        calc.data["alpha"] = _legacy_alpha_to_tensor(calc.data["raw_json"])

    calc.data["raman"] = {"polarization_frequencies": obj.polarization_frequencies,
                          "vibrational_frequencies": obj.vibrational_frequencies,
                          "polarizability_derivatives": obj.polarizability_derivatives,
                          "polarizability_derivatives_by_mode": obj.polarizability_derivatives,
                            "raman_by_freq": obj.raman_by_freq}


    # Keep other useful arrays (best-effort; may be None for some runs)
    calc.data["orbital_energies"] = obj.orbital_energies
    calc.data["hessian"] = obj.hessian
    calc.data["normal_modes"] = obj.normal_modes
    calc.data["vibrational_frequencies"] = obj.vibrational_frequencies
    calc.data["polarization_frequencies"] = getattr(obj, "polarization_frequencies", None)

    calc.data["molecule"] = obj.molecule
    calc.molecule = obj.molecule
    calc.meta["ground_state_energy"] = obj.ground_state_energy

    if obj.raman_by_freq is not None:
        for _, rows in obj.raman_by_freq.items():
            if isinstance(rows, list):
                rows.sort(key=lambda r: (float(r.get("freq_cm1", 0.0)), int(r.get("mode", 0))))
        calc.data["raman"] = {
            "polarization_frequencies": np.asarray(
                obj.polarization_frequencies, dtype=float
            )
            if obj.polarization_frequencies is not None
            else np.asarray([], dtype=float),
            "vibrational_frequencies": np.asarray(
                obj.vibrational_frequencies, dtype=float
            )
            if obj.vibrational_frequencies is not None
            else np.asarray([], dtype=float),
            "polarizability_derivatives": obj.polarizability_derivatives,
            "polarizability_derivatives_by_mode": obj.polarizability_derivatives_normal_modes,
            "raman_by_freq": obj.raman_by_freq,
        }

    if calc.molecule is None:
        calc.molecule = _load_molecule_from_input(calc.artifacts.get("input_json"))
        if calc.molecule is not None:
            calc.meta.setdefault("molecule_source", "input.json")
        else:
            calc.meta.setdefault("warnings", []).append(
                "MADNESS output missing molecule; input.json not found or invalid."
            )



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


def _load_molecule_from_input(path: Path | None):
    if path is None or not path.exists():
        return None
    data = _read_json(path)
    mol = data.get("molecule")
    if not isinstance(mol, dict):
        return None
    symbols = mol.get("symbols")
    geometry = mol.get("geometry")
    if symbols is None or geometry is None:
        return None

    import numpy as np
    import qcelemental as qcel

    units = mol.get("units") or mol.get("parameters", {}).get("units")
    coords = np.asarray(geometry, dtype=float)
    if isinstance(units, str) and units.lower() in ("bohr", "atomic", "au"):
        coords = coords * qcel.constants.bohr2angstroms
    symbols_sorted, coords_sorted = canonicalize_atom_order(list(symbols), coords, decimals=10)
    kwargs = {
        "symbols": symbols_sorted,
        "geometry": coords_sorted,
    }
    mol_charge = mol.get("charge")
    mol_mult = mol.get("multiplicity")
    if mol_charge is not None:
        kwargs["molecular_charge"] = mol_charge
    if mol_mult is not None:
        kwargs["molecular_multiplicity"] = mol_mult
    return qcel.models.Molecule(**kwargs)
