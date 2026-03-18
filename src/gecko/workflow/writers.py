"""Input file writers for MADNESS and DALTON calculations."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import qcelemental as qcel

Property = Literal["alpha", "beta", "raman"]
XCFunctional = str  # "hf", "b3lyp", "pbe0", etc.


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
        lines = ["dft"]
        if self.xc.lower() != "hf":
            lines.append(f"    xc {self.xc}")
        lines += [
            f"    k {self.k}",
            f"    econv {self.econv:.1e}",
            f"    dconv {self.dconv:.1e}",
            f"    maxiter {self.maxiter}",
            "    protocol [0.0001,1e-06,1e-7]",
            "end",
        ]
        return "\n".join(lines)

    def _molecule_section(self) -> str:
        geom = np.asarray(self.molecule.geometry).reshape(-1, 3)
        lines = [
            "molecule",
            "    eprec  1.0000e-06",
            "    units  atomic",
        ]
        for sym, coord in zip(self.molecule.symbols, geom):
            lines.append(f"    {sym:2s}  {coord[0]:20.10f}  {coord[1]:20.10f}  {coord[2]:20.10f}")
        lines.append("end")
        return "\n".join(lines)

    def _response_section(self) -> str:
        freq_str = "[" + ",".join(str(f) for f in self.frequencies) + "]"
        n_atoms = len(self.molecule.symbols)
        atom_indices = list(range(n_atoms))
        atom_idx_str = "[" + ",".join(str(i) for i in atom_indices) + "]"

        lines = ["response"]
        if self.property == "alpha":
            lines += [
                f"    dipole.frequencies {freq_str}",
                "    dipole.directions xyz",
                "    requested_properties [polarizability]",
            ]
        elif self.property == "beta":
            lines += [
                f"    dipole.frequencies {freq_str}",
                "    dipole.directions xyz",
                "    quadratic true",
                "    requested_properties [hyperpolarizability]",
            ]
        elif self.property == "raman":
            lines += [
                f"    dipole.frequencies {freq_str}",
                "    dipole.directions xyz",
                "    nuclear true",
                "    nuclear.directions xyz",
                "    nuclear.frequencies 0.0",
                f"    nuclear.atom_indices {atom_idx_str}",
                "    requested_properties [polarizability,raman]",
                "    property true",
            ]
        lines += [
            f"    maxiter {self.response_maxiter}",
            "end",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# DALTON
# ---------------------------------------------------------------------------


@dataclass
class DaltonInput:
    """Parameters for a DALTON response calculation.

    Generates a ``.dal`` input file and a ``.mol`` geometry file.

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
        Use direct integrals (``.DIRECT``).  Recommended for large systems.
    """

    molecule: qcel.models.Molecule
    mol_name: str
    basis: str
    xc: str = "hf"
    property: Property = "alpha"
    frequencies: list[float] = field(default_factory=lambda: [0.0])
    direct: bool = True

    def write(self, out_dir: Path) -> dict[str, Path]:
        """Write ``.dal`` and ``.mol`` files.

        Returns
        -------
        dict
            ``{"dal": Path, "mol": Path}``
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{self.mol_name}_{self.basis}"
        dal_path = out_dir / f"{stem}.dal"
        mol_path = out_dir / f"{stem}.mol"
        dal_path.write_text(self._render_dal())
        mol_path.write_text(self._render_mol())
        return {"dal": dal_path, "mol": mol_path}

    # ------------------------------------------------------------------

    def _render_dal(self) -> str:
        if self.property == "raman":
            return self._render_dal_raman()

        lines = ["**DALTON INPUT", ".RUN RESPONSE"]
        if self.direct:
            lines.append(".DIRECT")
        lines.append("**WAVE FUNCTIONS")
        if self.xc.lower() == "hf":
            lines.append(".HF")
        else:
            lines += [".DFT", self.xc.upper()]
        lines.extend(self._response_lines())
        lines.append("**END OF DALTON INPUT")
        return "\n".join(lines) + "\n"

    def _render_dal_raman(self) -> str:
        """Raman requires a PROPERTIES run (vibrational analysis)."""
        lines = ["**DALTON INPUT", ".RUN PROPERTIES"]
        if self.direct:
            lines.append(".DIRECT")
        lines.append("**WAVE FUNCTIONS")
        if self.xc.lower() == "hf":
            lines.append(".HF")
        else:
            lines += [".DFT", self.xc.upper()]
        lines += ["**PROPERTIES", ".VIBANA", ".RAMAN", "**END OF DALTON INPUT"]
        return "\n".join(lines) + "\n"

    def _response_lines(self) -> list[str]:
        lines = ["**RESPONSE"]
        if self.property == "alpha":
            lines += ["*LINEAR", ".ALPHA"]
        elif self.property == "beta":
            n = len(self.frequencies)
            freq_line = " ".join(str(f) for f in self.frequencies)
            lines += ["*QUADRA", ".DIPLEN", ".FREQUENCIES", str(n), freq_line]
        return lines

    def _render_mol(self) -> str:
        from gecko.plugins.dalton.legacy.dalton_write_inputs import to_string

        return to_string(self.molecule, self.basis, units="Angstrom") + "\n"


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

    Returns
    -------
    dict
        ``{"madness": [path, ...], "dalton": [path, ...]}``
    """
    out_dir = Path(out_dir)
    freqs = frequencies or [0.0]
    result: dict[str, list[Path]] = {"madness": [], "dalton": []}

    if "madness" in codes:
        mad_dir = out_dir / mol_name / "madness"
        inp = MadnessInput(
            molecule=molecule,
            mol_name=mol_name,
            property=property,
            frequencies=freqs,
            xc=xc,
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
