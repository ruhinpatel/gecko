from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import qcelemental as qcel

from gecko.molecule.canonical import canonicalize_atom_order


def find_task_by_type(json_data: dict, task_type: str) -> Optional[dict]:
    tasks = json_data.get("tasks", [])
    if not isinstance(tasks, list):
        return None
    for t in tasks:
        if isinstance(t, dict) and t.get("type") == task_type:
            return t
    return None


def extract_response_container(json_data: dict) -> Optional[dict]:
    """
    Return the container dict that holds response info.

    Supports:
      - legacy: task with type=="response"
      - MADQC: any task with properties.response_properties
      - top-level fallbacks
    """
    # legacy task style
    resp_task = find_task_by_type(json_data, "response")
    if isinstance(resp_task, dict):
        return resp_task

    # MADQC style task blocks (fallback search)
    tasks = json_data.get("tasks", [])
    if isinstance(tasks, list):
        for t in tasks:
            if not isinstance(t, dict):
                continue
            props = t.get("properties")
            if isinstance(props, dict) and isinstance(props.get("response_properties"), list):
                return t

    # top-level fallbacks
    if isinstance(json_data.get("properties"), dict) and isinstance(json_data["properties"].get("response_properties"), list):
        return json_data
    if isinstance(json_data.get("response_properties"), list):
        return json_data

    return None


def extract_response_properties(container: dict) -> Optional[list[dict]]:
    """
    Given a response container, return a list of response property records.

    Supports:
      - legacy: container["properties"] is already a list
      - MADQC: container["properties"]["response_properties"]
      - top-level: container["response_properties"]
    """
    props = container.get("properties")

    # legacy: list of records
    if isinstance(props, list):
        return props

    # MADQC: dict with response_properties
    if isinstance(props, dict) and isinstance(props.get("response_properties"), list):
        return props["response_properties"]

    # top-level fallback
    if isinstance(container.get("response_properties"), list):
        return container["response_properties"]

    return None


def _get_optional(container: dict[str, Any], *path: str) -> Any:
    cur: Any = container
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def to_qcel(mol: dict) -> qcel.models.Molecule:
    geometry = mol["geometry"]
    units = mol.get("parameters", {}).get("units", None)
    if units == "atomic":
        geometry = np.array(geometry) * qcel.constants.bohr2angstroms
    symbols_sorted, geometry_sorted = canonicalize_atom_order(
        mol["symbols"], geometry, decimals=10
    )
    return qcel.models.Molecule(symbols=symbols_sorted, geometry=geometry_sorted)


def tensor_to_numpy(tensor: dict) -> np.ndarray:
    return np.array(tensor["vals"]).reshape(tensor["dims"])


def normalize_ijk_component(comp) -> str:
    if comp is None:
        raise ValueError("component is None")

    if isinstance(comp, str):
        s = comp.strip()
        # MADQC-style tokens in a list sometimes get stringified; handle loosely
        if "Dipole_" in s or "_" in s:
            parts = [p.strip() for p in s.replace(",", " ").split()]
            comp = parts
        else:
            return s.lower()

    tokens = list(comp)

    def tok_to_axis(t) -> str:
        t = str(t).strip()
        if "_" in t:
            t = t.split("_")[-1]
        t = t.lower()
        if t not in ("x", "y", "z"):
            raise ValueError(f"Unrecognized component token: {t!r}")
        return t

    return "".join(tok_to_axis(t) for t in tokens)


class madqc_parser:
    def __init__(self, json_file: Path | str):
        self.json_file = Path(json_file)

        self.ground_state_energy: Optional[float] = None
        self.molecule: Optional[qcel.models.Molecule] = None
        self.orbital_energies: Optional[np.ndarray] = None

        self.alpha_pivot: Optional[pd.DataFrame] = None
        self.beta_pivot: Optional[pd.DataFrame] = None
        self.raman_pivot: Optional[pd.DataFrame] = None

        self.vibrational_frequencies: Optional[np.ndarray] = None
        self.hessian: Optional[np.ndarray] = None
        self.normal_modes: Optional[np.ndarray] = None

        self.polarization_frequencies: Optional[np.ndarray] = None
        self.raman_by_freq: Optional[dict[float, list[dict]]] = None

        self.polarizability_derivatives: list[np.ndarray] = []
        self.polarizability_derivatives_normal_modes: list[np.ndarray] = []

        self.warnings: list[str] = []

        self.read_data()

    def read_data(self) -> None:
        with self.json_file.open("r", encoding="utf-8") as f:
            json_data = json.load(f)

        # -------------------------
        # SCF: support both styles
        # -------------------------
        scf_task = find_task_by_type(json_data, "scf")
        if isinstance(scf_task, dict):
            # legacy: energy directly on task
            if "energy" in scf_task:
                self.ground_state_energy = scf_task.get("energy")

        # MADQC-style: tasks[*].scf block
        # Try to find molecule + eigenvalues if present (optional)
        try:
            tasks = json_data.get("tasks", [])
            if isinstance(tasks, list):
                for t in tasks:
                    if not isinstance(t, dict):
                        continue
                    scf = t.get("scf")
                    if isinstance(scf, dict):
                        # molecule
                        mol = scf.get("molecule")
                        if isinstance(mol, dict) and self.molecule is None:
                            self.molecule = to_qcel(mol)
                        # energy
                        props = scf.get("properties")
                        if isinstance(props, dict) and self.ground_state_energy is None and "energy" in props:
                            self.ground_state_energy = props["energy"]
                        # eigenvalues
                        if "scf_eigenvalues_a" in scf and self.orbital_energies is None:
                            self.orbital_energies = tensor_to_numpy(scf["scf_eigenvalues_a"])
        except Exception as e:
            self.warnings.append(f"Failed MADQC scf parse: {type(e).__name__}: {e}")

        # --------------------------------
        # Response properties (optional)
        # --------------------------------
        resp_container = extract_response_container(json_data)
        if resp_container is None:
            self.warnings.append("No response container found.")
            return

        rprops_raw = extract_response_properties(resp_container)
        if not rprops_raw:
            self.warnings.append("Response container found, but no response properties list.")
            return

        rprops = pd.DataFrame(rprops_raw)

        # ------------------------
        # Polarizability alpha
        # ------------------------
        try:
            alpha = rprops.query("property == 'polarizability'").copy()
            if len(alpha) > 0:
                # legacy uses "xx", MADQC might be ["x","x"]-like
                alpha["ij"] = alpha.component.apply(
                    lambda x: "".join(x).lower() if not isinstance(x, str) else x.lower()
                )
                alpha["omega"] = alpha.freqB.astype(float)
                alpha = alpha[["omega", "ij", "value"]]
                alpha_pivot = alpha.pivot(index="omega", columns="ij", values="value")
                if all(c in alpha_pivot.columns for c in ("xx", "yy", "zz")):
                    alpha_pivot["mean_alpha"] = (alpha_pivot["xx"] + alpha_pivot["yy"] + alpha_pivot["zz"]) / 3.0
                self.alpha_pivot = alpha_pivot
        except Exception as e:
            self.warnings.append(f"Failed to parse alpha: {type(e).__name__}: {e}")

        # ------------------------
        # Hyperpolarizability beta
        # ------------------------
        try:
            beta = rprops.query("property == 'hyperpolarizability'").copy()
            if len(beta) > 0:
                beta["ijk"] = beta.component.apply(normalize_ijk_component)
                beta["omegaB"] = beta.freqB.astype(float)
                beta["omegaC"] = beta.freqC.astype(float)
                beta["omegaA"] = -(beta["omegaB"] + beta["omegaC"])
                beta = beta[["omegaA", "omegaB", "omegaC", "ijk", "value"]]
                self.beta_pivot = beta.pivot_table(
                    index=["omegaA", "omegaB", "omegaC"], columns="ijk", values="value"
                )
        except Exception as e:
            self.warnings.append(f"Failed to parse beta: {type(e).__name__}: {e}")

        # -------------
        # Raman (optional)
        # -------------
        try:
            raman = rprops.query("property == 'raman'").copy()
            if len(raman) > 0:
                def _ij(comp):
                    if isinstance(comp, str):
                        return comp.lower()
                    return "".join([str(v).split("_")[-1].lower() for v in comp[0:2]])

                def _dnuc(comp):
                    if isinstance(comp, str):
                        return comp
                    return "".join([str(v).split("_")[-1] for v in comp[2:]])

                raman["ij"] = raman.component.apply(_ij)
                raman["dnuc"] = raman.component.apply(_dnuc)
                raman["omega"] = raman.freqB.astype(float)
                raman = raman[["omega", "ij", "dnuc", "value"]]
                self.raman_pivot = raman.pivot_table(index=["omega", "dnuc"], columns=["ij"], values="value")
        except Exception as e:
            self.warnings.append(f"Failed to parse raman response: {type(e).__name__}: {e}")

        # ------------------------------------------------
        # Vib/Raman spectra (optional; MADQC-style only)
        # ------------------------------------------------
        vib = _get_optional(resp_container, "properties", "vibrational_analysis")
        if isinstance(vib, dict):
            try:
                if "hessian" in vib:
                    self.hessian = tensor_to_numpy(vib["hessian"])
                if "normalmodes" in vib:
                    self.normal_modes = tensor_to_numpy(vib["normalmodes"])
            except Exception as e:
                self.warnings.append(f"Failed vib parse: {type(e).__name__}: {e}")

        raman_spectra = _get_optional(resp_container, "properties", "raman_spectra")
        if isinstance(raman_spectra, dict):
            try:
                if "polarization_frequencies" in raman_spectra:
                    self.polarization_frequencies = np.array(
                        raman_spectra["polarization_frequencies"], dtype=float
                    )
                if "vibrational_frequencies" in raman_spectra:
                    self.vibrational_frequencies = np.array(
                        raman_spectra["vibrational_frequencies"], dtype=float
                    )

                rs = raman_spectra.get("raman_spectra")
                if isinstance(rs, dict):
                    self.raman_by_freq = {float(k): v for k, v in rs.items()}

                self.polarizability_derivatives = []
                self.polarizability_derivatives_normal_modes = []

                pd_list = raman_spectra.get("polarizability_derivatives")
                pd_nm_list = raman_spectra.get("polarizability_derivatives_normal_modes")

                if isinstance(pd_list, list):
                    for item in pd_list:
                        self.polarizability_derivatives.append(tensor_to_numpy(item))

                if isinstance(pd_nm_list, list):
                    for item in pd_nm_list:
                        self.polarizability_derivatives_normal_modes.append(tensor_to_numpy(item))
            except Exception as e:
                self.warnings.append(f"Failed raman_spectra parse: {type(e).__name__}: {e}")
