from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import qcelemental as qcel

BLANK_RE = re.compile(r"^\s*$")
MOLECULAR_GEOMETRY_RE = re.compile(r"^\s*Molecular geometry\s*\(au\)\s*$", re.IGNORECASE)
POLAR_FREQ_TITLE_RE = re.compile(r"^\s*\++\s*Frequency dependent polarizabilities\s*\++\s*$", re.IGNORECASE)
POLAR_TENSOR_HDR_RE = re.compile(
    r"^\s*Polarizability tensor for frequency\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*au\s*$"
)
ISO_POLAR_RE = re.compile(
    r"^\s*Isotropic polarizability:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?")
HF_ENERGY_RE = re.compile(r"^@\s*Final HF energy:\s*(-?\d+\.\d+)\s*$")


def parse_last_molecular_geometry(
    lines: Sequence[str],
    *,
    header_skip: int = 3,
) -> qcel.models.Molecule:
    last_header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if MOLECULAR_GEOMETRY_RE.match(line):
            last_header_idx = i

    if last_header_idx is None:
        raise ValueError("No 'Molecular geometry (au)' header found.")

    start = last_header_idx + header_skip
    atoms: List[str] = []
    coords: List[List[float]] = []

    for j in range(start, len(lines)):
        line = lines[j]
        if BLANK_RE.match(line):
            break
        parts = line.split()
        if len(parts) < 4:
            break
        label = parts[0].split("_", 1)[0]
        start_float_index = 2 if len(parts) == 5 else 1
        x, y, z = (
            float(parts[start_float_index]),
            float(parts[start_float_index + 1]),
            float(parts[start_float_index + 2]),
        )
        atoms.append(label)
        coords.append([x, y, z])

    if not atoms:
        raise ValueError("Found geometry header but no atom lines followed.")

    bohr_to_ang = qcel.constants.bohr2angstroms
    coords = [[c * bohr_to_ang for c in vec] for vec in coords]
    return qcel.models.Molecule(symbols=atoms, geometry=coords)


def parse_frequency_polarizability_tensors(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[Dict[float, Dict[str, Any]], int]:
    i = start_index
    pol_by_freq: Dict[float, Dict[str, Any]] = {}
    nlines = len(lines)

    while i < nlines:
        m_hdr = POLAR_TENSOR_HDR_RE.match(lines[i])
        if not m_hdr:
            i += 1
            continue

        freq_au = float(m_hdr.group(1))
        i += 1

        while i < nlines and "Ex" not in lines[i]:
            i += 1
        if i >= nlines:
            break

        i += 1
        while i < nlines and not lines[i].strip():
            i += 1

        tensor = np.zeros((3, 3), dtype=float)
        for row in range(3):
            if i >= nlines:
                raise ValueError("Unexpected end of file while reading tensor rows.")
            line = lines[i]
            nums = FLOAT_RE.findall(line)
            if len(nums) < 3:
                raise ValueError(f"Could not parse three floats from tensor row: {line!r}")
            vals = [float(x) for x in nums[-3:]]
            tensor[row, :] = vals
            i += 1

        iso_val: Optional[float] = None
        while i < nlines:
            line = lines[i]
            m_iso = ISO_POLAR_RE.match(line)
            if m_iso:
                iso_val = float(m_iso.group(1))
                i += 1
                break
            if POLAR_TENSOR_HDR_RE.match(line):
                break
            i += 1

        pol_by_freq[freq_au] = {"tensor": tensor, "iso": iso_val}

    return pol_by_freq, i


def parse_polarizability_section(
    lines: Sequence[str],
    start_index: int = 0,
    nfreqs: Optional[int] = None,
) -> Tuple[np.ndarray, List[float], int]:
    last_header_idx: Optional[int] = None
    for i in range(start_index, len(lines)):
        if POLAR_FREQ_TITLE_RE.match(lines[i]):
            last_header_idx = i

    if last_header_idx is None:
        raise ValueError("No 'Frequency dependent polarizabilities' section found.")

    pol_by_freq, next_idx = parse_frequency_polarizability_tensors(
        lines, start_index=last_header_idx
    )
    if not pol_by_freq:
        raise ValueError("Found polarizability header but no tensors.")

    freqs_sorted = sorted(pol_by_freq.keys())
    if nfreqs is not None:
        freqs_sorted = freqs_sorted[:nfreqs]

    tensors = np.array([pol_by_freq[f]["tensor"] for f in freqs_sorted])
    return tensors, freqs_sorted, next_idx


def parse_hf_energy(lines: Sequence[str]) -> float:
    for line in lines:
        m = HF_ENERGY_RE.match(line.strip())
        if m:
            return float(m.group(1))
    raise ValueError("Final HF energy line not found.")


class DaltonParser:
    def __init__(self, src: str | Path | Sequence[str]):
        if isinstance(src, (str, Path)):
            self.lines = Path(src).read_text(encoding="utf-8", errors="ignore").splitlines()
        else:
            self.lines = list(src)

        self.molecule: Optional[qcel.models.Molecule] = None
        self.polarizability: Optional[np.ndarray] = None
        self.hf_energy: Optional[float] = None

    def parse_geometry(self) -> qcel.models.Molecule:
        self.molecule = parse_last_molecular_geometry(self.lines)
        return self.molecule

    def parse_polarizability(self) -> np.ndarray:
        tensors, _freqs, _ = parse_polarizability_section(self.lines, start_index=0)
        self.polarizability = tensors
        return tensors

    def parse_final_hf_energy(self) -> float:
        self.hf_energy = parse_hf_energy(self.lines)
        return self.hf_energy
