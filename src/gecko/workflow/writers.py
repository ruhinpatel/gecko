"""Input file writers for MADNESS and DALTON calculations."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import qcelemental as qcel

from gecko.workflow.params import (
    DFTParams,
    MoleculeParams,
    ResponseParams,
    _RESPONSE_FIELD_MAP,
    _render_value,
)

Property = Literal["alpha", "beta", "raman"]
XCFunctional = str  # "hf", "b3lyp", "pbe0", etc.

# Mapping from DFTParams / MoleculeParams field names to MADNESS keys.
# Only entries that differ from the Python attribute name are listed.
_DFT_FIELD_MAP: dict[str, str] = {
    "spin_restricted": "spin_restricted",
}
_MOL_FIELD_MAP: dict[str, str] = {
    "no_orient": "no_orient",
}


# ---------------------------------------------------------------------------
# MADNESS
# ---------------------------------------------------------------------------


@dataclass
class MadnessInput:
    """Parameters for a MADNESS response calculation.

    Generates a single ``.in`` file with embedded ``molecule`` block.

    Parameters
    ----------
    molecule : qcel.models.Molecule
        Molecular geometry (coordinates in Bohr).
    mol_name : str
        Short identifier used for the output filename, e.g. ``"SO2"``.
    xc : str
        Exchange-correlation functional.  Use ``"hf"`` for Hartree-Fock.
    k : int
        Wavelet order (6–10 are typical).
    econv : float
        Energy convergence threshold.
    dconv : float
        Density convergence threshold.
    maxiter : int
        Maximum SCF iterations.
    response_maxiter : int
        Maximum response iterations.
    property : {"alpha", "beta", "raman"}
        Property to compute.
    frequencies : list[float]
        Optical frequencies in atomic units (Hartree).
    dft_params : DFTParams, optional
        Fine-grained overrides for the ``dft`` section.  Any non-None
        field is merged on top of the defaults derived from the
        high-level parameters above.
    molecule_params : MoleculeParams, optional
        Fine-grained overrides for the ``molecule`` section.
    response_params : ResponseParams, optional
        Fine-grained overrides for the ``response`` section.
    """

    molecule: qcel.models.Molecule
    mol_name: str
    xc: str = "hf"
    k: int = 8
    econv: float = 1e-6
    dconv: float = 1e-4
    maxiter: int = 25
    response_maxiter: int = 25
    property: Property = "alpha"
    frequencies: list[float] = field(default_factory=lambda: [0.0])
    # Fine-grained overrides (all optional; None = use high-level defaults)
    dft_params: Optional[DFTParams] = None
    molecule_params: Optional[MoleculeParams] = None
    response_params: Optional[ResponseParams] = None

    def write(self, out_dir: Path) -> Path:
        """Write the ``.in`` file and return its path."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.property}_{self.mol_name}".lower()
        in_path = out_dir / f"{stem}.in"
        in_path.write_text(self._render())
        return in_path

    # ------------------------------------------------------------------

    def _render(self) -> str:
        sections = [
            self._dft_section(),
            "",
            self._molecule_section(),
            "",
            self._response_section(),
        ]
        return "\n".join(sections) + "\n"

    def _dft_section(self) -> str:
        # Phase 1: high-level defaults
        base: dict[str, object] = {
            "k": self.k,
            "econv": self.econv,
            "dconv": self.dconv,
            "maxiter": self.maxiter,
            "protocol": [0.0001, 1e-6, 1e-7],
        }
        if self.xc.lower() != "hf":
            base["xc"] = self.xc

        # Phase 2: overlay from DFTParams
        if self.dft_params is not None:
            for f in dataclasses.fields(self.dft_params):
                v = getattr(self.dft_params, f.name)
                if v is None:
                    continue
                key = _DFT_FIELD_MAP.get(f.name, f.name)
                if key == "xc":
                    if str(v).lower() == "hf":
                        base.pop("xc", None)
                    else:
                        base["xc"] = v
                else:
                    base[key] = v

        lines = ["dft"]
        for key, val in base.items():
            lines.append(f"    {key} {_render_value(val)}")
        lines.append("end")
        return "\n".join(lines)

    def _molecule_section(self) -> str:
        # Phase 1: defaults
        base: dict[str, object] = {
            "eprec": 1e-6,
            "units": "atomic",
        }

        # Phase 2: overlay from MoleculeParams
        if self.molecule_params is not None:
            for f in dataclasses.fields(self.molecule_params):
                v = getattr(self.molecule_params, f.name)
                if v is None:
                    continue
                key = _MOL_FIELD_MAP.get(f.name, f.name)
                base[key] = v

        geom = np.asarray(self.molecule.geometry).reshape(-1, 3)
        lines = ["molecule"]
        for key, val in base.items():
            lines.append(f"    {key}  {_render_value(val)}")
        for sym, coord in zip(self.molecule.symbols, geom):
            lines.append(
                f"    {sym:2s}  {coord[0]:20.10f}  {coord[1]:20.10f}  {coord[2]:20.10f}"
            )
        lines.append("end")
        return "\n".join(lines)

    def _response_section(self) -> str:
        n_atoms = len(self.molecule.symbols)

        # Phase 1: property-based presets
        base: dict[str, object] = {}
        if self.property == "alpha":
            base["dipole.frequencies"] = self.frequencies
            base["dipole.directions"] = "xyz"
            base["requested_properties"] = ["polarizability"]
        elif self.property == "beta":
            base["dipole.frequencies"] = self.frequencies
            base["dipole.directions"] = "xyz"
            base["quadratic"] = True
            base["requested_properties"] = ["hyperpolarizability"]
        elif self.property == "raman":
            base["dipole.frequencies"] = self.frequencies
            base["dipole.directions"] = "xyz"
            base["nuclear"] = True
            base["nuclear.directions"] = "xyz"
            base["nuclear.frequencies"] = 0.0
            base["nuclear.atom_indices"] = list(range(n_atoms))
            base["requested_properties"] = ["polarizability", "raman"]
            base["property"] = True

        base["maxiter"] = self.response_maxiter

        # Phase 2: overlay from ResponseParams
        if self.response_params is not None:
            for f in dataclasses.fields(self.response_params):
                v = getattr(self.response_params, f.name)
                if v is None:
                    continue
                key = _RESPONSE_FIELD_MAP.get(f.name, f.name)
                base[key] = v

        lines = ["response"]
        for key, val in base.items():
            lines.append(f"    {key} {_render_value(val)}")
        lines.append("end")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DALTON
# ---------------------------------------------------------------------------


@dataclass
class DaltonInput:
    """Parameters for a DALTON response calculation.

    Generates a ``.dal`` input file and a ``.mol`` geometry file.
    For Raman calculations, also generates an ``optimize.dal`` file
    (geometry optimisation must be run before the Raman step).

    Input files are generated via ``daltonproject.dalton`` rather than
    custom renderers.  The ``direct`` field is kept for API compatibility
    but has no effect (DaltonProject controls integral handling).

    Parameters
    ----------
    molecule : qcel.models.Molecule
        Molecular geometry (coordinates in Bohr).
    mol_name : str
        Short identifier used for output filenames, e.g. ``"SO2"``.
    basis : str
        Basis set name (e.g. ``"aug-cc-pVDZ"``).
    xc : str
        Exchange-correlation functional.  Use ``"hf"`` for Hartree-Fock.
    property : {"alpha", "beta", "raman"}
        Property to compute.
    frequencies : list[float]
        Optical frequencies in atomic units (Hartree).
    direct : bool
        Kept for backward compatibility; has no effect.
    multiplicity : int
        Spin multiplicity (1 = singlet/closed-shell, 2 = doublet, etc.).
    rohf_closed : int, optional
        Number of closed-shell orbitals for ROHF open-shell calculations.
        Required when ``multiplicity > 1``.
    rohf_open : int, optional
        Number of singly-occupied orbitals for ROHF open-shell calculations.
        Required when ``multiplicity > 1``.
    """

    molecule: qcel.models.Molecule
    mol_name: str
    basis: str
    xc: str = "hf"
    property: Property = "alpha"
    frequencies: list[float] = field(default_factory=lambda: [0.0])
    direct: bool = True
    multiplicity: int = 1
    rohf_closed: Optional[int] = None
    rohf_open: Optional[int] = None

    def write(self, out_dir: Path) -> dict[str, Path]:
        """Write input files and return a dict of ``{label: Path}``.

        For alpha/beta: ``{"dal": ..., "mol": ...}``
        For raman:      ``{"optimize_dal": ..., "raman_dal": ..., "mol": ...}``
        """
        import daltonproject as dp
        from daltonproject.dalton.program import dalton_input, molecule_input

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.mol_name}_{self.basis}"

        # --- Build DaltonProject objects ---
        bohr2ang = qcel.constants.conversion_factor("bohr", "angstrom")
        geom = np.asarray(self.molecule.geometry).reshape(-1, 3) * bohr2ang
        atoms_str = "; ".join(
            f"{sym} {x:.10f} {y:.10f} {z:.10f}"
            for sym, (x, y, z) in zip(self.molecule.symbols, geom)
        )
        charge = int(round(self.molecule.molecular_charge))
        dp_mol = dp.Molecule(atoms=atoms_str, charge=charge, multiplicity=self.multiplicity)
        dp_basis = dp.Basis(basis=self.basis)

        method_name = "HF" if self.xc.lower() == "hf" else self.xc.upper()
        dp_method = dp.QCMethod(method_name)
        if self.multiplicity > 1 and self.rohf_closed is not None and self.rohf_open is not None:
            dp_method.scf_occupation(
                closed_shells=[self.rohf_closed],
                open_shells=[self.rohf_open],
            )

        # --- Write .mol file ---
        mol_path = out_dir / f"{stem}.mol"
        mol_path.write_text(molecule_input(dp_mol, dp_basis))

        # --- Write .dal file(s) ---
        if self.property == "raman":
            # TODO(BTS-75): validate DaltonProject Raman (.WALK) workflow before replacing
            opt_path = out_dir / "optimize.dal"
            ram_path = out_dir / "raman.dal"
            opt_path.write_text(self._render_dal_optimize())
            ram_path.write_text(self._render_dal_raman())
            return {"optimize_dal": opt_path, "raman_dal": ram_path, "mol": mol_path}

        freqs = [float(f) for f in self.frequencies]
        if self.property == "alpha":
            dp_prop = dp.Property(polarizabilities={"frequencies": freqs})
        else:  # beta
            dp_prop = dp.Property(first_hyperpolarizability=True)

        dal_path = out_dir / f"{stem}.dal"
        dal_path.write_text(dalton_input(dp_method, dp_prop, dp_mol))
        return {"dal": dal_path, "mol": mol_path}

    # ------------------------------------------------------------------
    # Raman legacy renderers — kept until DaltonProject .WALK workflow
    # is validated (TODO BTS-75).

    def _wf_lines(self) -> list[str]:
        if self.xc.lower() == "hf":
            return ["**WAVE FUNCTIONS", ".HF"]
        return ["**WAVE FUNCTIONS", ".DFT", self.xc.upper()]

    def _render_dal_optimize(self) -> str:
        lines = [
            "**DALTON INPUT",
            ".OPTIMIZE",
            "**OPTIMIZE",
            ".2NDORD",
        ]
        lines.extend(self._wf_lines())
        lines += [
            "*SCF INPUT",
            ".THRESH",
            "1.0D-8",
            "**PROPERTIES",
            ".VIBANA",
            ".SHIELD",
            "**VIBANA",
            ".PRINT",
            "100",
            "**END OF DALTON INPUT",
        ]
        return "\n".join(lines) + "\n"

    def _render_dal_raman(self) -> str:
        n = len(self.frequencies)
        freq_line = " ".join(str(f) for f in self.frequencies)
        abalnr_block = [
            "*ABALNR",
            ".THRESH",
            "1.0D-7",
            ".FREQUE",
            str(n),
            freq_line,
        ]
        lines = ["**DALTON INPUT", ".WALK", "*WALK", ".NUMERI"]
        lines.extend(self._wf_lines())
        lines += ["*SCF INPUT", ".THRESH", "1.0D-8", "**START", ".RAMAN"]
        lines.extend(abalnr_block)
        lines += ["**EACH STEP", ".RAMAN"]
        lines.extend(abalnr_block)
        lines += [
            "**PROPERTIES", ".RAMAN", ".VIBANA",
            "*RESPONSE", ".THRESH", "1.0D-6",
        ]
        lines.extend(abalnr_block)
        lines += ["*VIBANA", ".PRINT", "100", "**END OF DALTON INPUT"]
        return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Convenience: generate a full calculation directory
# ---------------------------------------------------------------------------


def generate_calc_dir(
    molecule: qcel.models.Molecule,
    mol_name: str,
    property: Property,
    *,
    codes: list[Literal["madness", "dalton"]],
    basis_sets: list[str],
    frequencies: list[float] | None = None,
    xc: str = "hf",
    out_dir: Path,
    dft_params: Optional[DFTParams] = None,
    molecule_params: Optional[MoleculeParams] = None,
    response_params: Optional[ResponseParams] = None,
) -> dict[str, list[Path]]:
    """Generate input files for one or more codes and basis sets.

    Parameters
    ----------
    molecule : qcel.models.Molecule
        Molecular geometry.
    mol_name : str
        Short molecule identifier, e.g. ``"SO2"``.
    property : {"alpha", "beta", "raman"}
        Property to compute.
    codes : list
        Which codes to generate inputs for: ``["madness"]``, ``["dalton"]``,
        or ``["madness", "dalton"]``.
    basis_sets : list[str]
        Basis set names for DALTON runs (ignored for MADNESS which uses MRA).
    frequencies : list[float] or None
        Optical frequencies in Hartree.  Defaults to ``[0.0]``.
    xc : str
        Exchange-correlation functional.
    out_dir : Path
        Root output directory.  Sub-directories are created automatically.
    dft_params : DFTParams, optional
        Fine-grained overrides for the MADNESS ``dft`` section.
    molecule_params : MoleculeParams, optional
        Fine-grained overrides for the MADNESS ``molecule`` section.
    response_params : ResponseParams, optional
        Fine-grained overrides for the MADNESS ``response`` section.

    Returns
    -------
    dict
        ``{"madness": [path, ...], "dalton": [path, ...]}``
    """
    out_dir = Path(out_dir)
    freqs = frequencies or [0.0]
    result: dict[str, list[Path]] = {"madness": [], "dalton": []}

    if "madness" in codes:
        mad_dir = out_dir / mol_name / (f"mad-{tier}" if tier else "madness")
        inp = MadnessInput(
            molecule=molecule,
            mol_name=mol_name,
            property=property,
            frequencies=freqs,
            xc=xc,
            dft_params=dft_params,
            molecule_params=molecule_params,
            response_params=response_params,
        )
        path = inp.write(mad_dir)
        result["madness"].append(path)

    if "dalton" in codes:
        for basis in basis_sets:
            dal_dir = out_dir / mol_name / "dalton" / basis
            inp = DaltonInput(
                molecule=molecule,
                mol_name=mol_name,
                basis=basis,
                property=property,
                frequencies=freqs,
                xc=xc,
            )
            paths = inp.write(dal_dir)
            result["dalton"].extend(paths.values())

    return result
