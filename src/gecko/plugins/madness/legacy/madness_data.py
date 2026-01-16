# workflow/raman/madness_data.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import qcelemental as qcel
#from quantumresponsepro import MADMolecule  # as you already had

def to_qcel(mol: dict) -> qcel.models.Molecule:
    """
    Convert a MADNESS-style molecule dict to a qcelemental Molecule.
    """

    geometry =mol["geometry"]
    # print(qcel.constants.bohr2angstroms)
    # print(geometry)
    units=mol['parameters']['units']
    if units =="atomic":
        geometry = np.array(geometry) * qcel.constants.bohr2angstroms

    return qcel.models.Molecule(
        symbols=mol["symbols"],
        geometry=geometry,
    )


def tensor_to_numpy(tensor: dict) -> np.ndarray:
    """
    Convert a MADNESS tensor stored as a dict with 'vals' and 'dims'
    into a NumPy array.
    """
    return np.array(tensor["vals"]).reshape(tensor["dims"])

def normalize_ijk_component(comp) -> str:
    """
    Normalize hyperpolarizability component labels to 'xyz' style.

    Supports:
      - ['Dipole_X','Dipole_Y','Dipole_Z']  -> 'xyz'
      - ['X','Y','Z'] or ['x','y','z']      -> 'xyz'
      - 'xyz'                               -> 'xyz'
      - ('Dipole_X','Dipole_X','Dipole_Y')  -> 'xxy'
    """
    if comp is None:
        raise ValueError("component is None")

    # Already a compact string like "xxy"
    if isinstance(comp, str):
        s = comp.strip()
        # handle "Dipole_X,Dipole_Y,..." as a string (just in case)
        if "Dipole_" in s or "_" in s:
            parts = [p.strip() for p in s.replace(",", " ").split()]
            comp = parts
        else:
            # assume already xyz-like
            return s.lower()

    # Now treat as iterable of tokens
    tokens = list(comp)

    def tok_to_axis(t) -> str:
        t = str(t).strip()
        # e.g., "Dipole_X" -> "X"
        if "_" in t:
            t = t.split("_")[-1]
        t = t.lower()
        # allow x/y/z only
        if t not in ("x", "y", "z"):
            raise ValueError(f"Unrecognized component token: {t!r}")
        return t

    return "".join(tok_to_axis(t) for t in tokens)


class madqc_parser:
    """
    Container for MADNESS JSON output, with convenient access to:

      - ground_state_energy
      - molecule (qcelemental.Molecule)
      - orbital_energies
      - polarizability (alpha_pivot, DataFrame)
      - hyperpolarizability (beta_pivot, DataFrame)
      - Raman properties (raman_pivot, etc.)
      - vibrational frequencies, Hessian, normal modes
      - Raman spectra-derived quantities
    """

    def __init__(self, json_file: Path | str):
        self.json_file = Path(json_file)
        # initialized here for clarity
        self.ground_state_energy: Optional[float] = None
        self.molecule: Optional[qcel.models.Molecule] = None
        self.orbital_energies: Optional[np.ndarray] = None

        self.alpha_pivot: Optional[pd.DataFrame] = None
        self.beta_pivot: Optional[pd.DataFrame] = None
        self.raman_pivot: Optional[pd.DataFrame] = None

        self.vibrational_frequencies: Optional[np.ndarray] = None
        self.hessian: Optional[np.ndarray] = None
        self.normal_modes: Optional[np.ndarray] = None

        self.raman_by_freq: dict[float, list[dict]] = None

        self.polarizability_derivatives: list[np.ndarray] = []
        self.polarizability_derivatives_normal_modes: list[np.ndarray] = []

        self.read_data()

    def read_data(self) -> None:
        with self.json_file.open("r") as f:
            json_data = json.load(f)

        try:
            # --- SCF & orbitals ---
            scf_data = json_data["tasks"][0]["scf"]
            mol = scf_data["molecule"]
            self.ground_state_energy = scf_data["properties"]["energy"]
            self.molecule = to_qcel(mol)
            self.orbital_energies = tensor_to_numpy(scf_data["scf_eigenvalues_a"])

            # --- Response properties ---
            rdata = json_data["tasks"][1]
            rprops = pd.DataFrame(rdata["properties"]["response_properties"])

            # Polarizability α(ω)
            alpha = rprops.query("property == 'polarizability'").copy()
            alpha["ij"] = alpha.component.apply(lambda x: "".join(x))
            alpha["omega"] = alpha.freqB.astype(float)
            alpha = alpha[["omega", "ij", "value"]]
            alpha_pivot = alpha.pivot(index="omega", columns="ij", values="value")
            alpha_pivot["mean_alpha"] = (
                alpha_pivot["xx"] + alpha_pivot["yy"] + alpha_pivot["zz"]
            ) / 3.0
            self.alpha_pivot = alpha_pivot

            # Hyperpolarizability β(ωA; ωB, ωC)
            beta = rprops.query("property == 'hyperpolarizability'").copy()
            beta["ijk"] = beta.component.apply(normalize_ijk_component)

            beta["omegaB"] = beta.freqB
            beta["omegaC"] = beta.freqC
            beta["omegaA"] = -(beta.freqB + beta.freqC)
            beta = beta[["omegaA", "omegaB", "omegaC", "ijk", "value"]]
            self.beta_pivot = beta.pivot_table(
                index=["omegaA", "omegaB", "omegaC"], columns="ijk", values="value"
            )

            # Raman response
            raman = rprops.query("property == 'raman'").copy()
            raman["ij"] = raman.component.apply(
                lambda x: "".join([v.split("_")[1] for v in x[0:2]])
            )
            raman["dnuc"] = raman.component.apply(
                lambda x: "".join([v.split("_")[1] for v in x[2:]])
            )
            raman["omega"] = raman.freqB
            raman = raman[["omega", "ij", "dnuc", "value"]]
            self.raman_pivot = raman.pivot_table(
                index=["omega", "dnuc"], columns=["ij"], values="value"
            )

            # Vibrational analysis
            vib_data = rdata["properties"]["vibrational_analysis"]

            self.hessian = tensor_to_numpy(vib_data["hessian"])
            self.normal_modes = tensor_to_numpy(vib_data["normalmodes"])

            # Raman spectra block
            raman_spectra = rdata["properties"]["raman_spectra"]
            self.polarization_frequencies = np.array(raman_spectra["polarization_frequencies"])
            self.vibrational_frequencies = np.array(
                raman_spectra["vibrational_frequencies"]
            )

            self.polarizability_derivatives = []
            self.polarizability_derivatives_normal_modes = []

            raman_related_properties= raman_spectra["raman_spectra"]
            self.raman_by_freq = {}
            for freq_, item in raman_related_properties.items():
                freq = float(freq_)
                self.raman_by_freq[freq] = item

            for i, _freq in enumerate(self.polarization_frequencies):
                self.polarizability_derivatives.append(
                    tensor_to_numpy(raman_spectra["polarizability_derivatives"][i])
                )
                self.polarizability_derivatives_normal_modes.append(
                    tensor_to_numpy(
                        raman_spectra["polarizability_derivatives_normal_modes"][i]
                    )
                )

        except Exception as e:
            print("No response data found:", e)
