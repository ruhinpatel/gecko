# dalton_parser.py
from __future__ import annotations

import sys
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import qcelemental as qcel

# ---------- Shared regexes ----------
BLANK_RE = re.compile(r"^\s*$")
SEP_RE = re.compile(r"^\s*[-=]{5,}\s*$", re.IGNORECASE)

# Geometry
MOLECULAR_GEOMETRY_RE = re.compile(
    r"^\s*Molecular geometry\s*\(au\)\s*$", re.IGNORECASE
)

# Optimized Geometry
OPTIMIZED_GEOMETRY_RE = re.compile(r"^\s*Final geometry\s*\(au\)\s*$", re.IGNORECASE)

# Column headers like: "Column   1     Column   2 ... (≤6 per block)"
COL_HEADER_RE = re.compile(r"^\s*Column\s+\d+(?:\s+Column\s+\d+){0,5}\s*$")
END_MATRIX_RE = re.compile(r"^\s*====\s*End of matrix output\s*====\s*$", re.IGNORECASE)

# Sections
HESSIAN_HDR_RE = re.compile(r"^\s*Hessian in non-symmetry coordinates\s*$")
NORMAL_COORDS_HDR_RE = re.compile(r"^\s*Normal coordinates in Cartesian basis\s*$")
MWH_EIGEN_HDR_RE = re.compile(
    r"^\s*Eigenvalues of mass-weighted Hessian\s*$", re.IGNORECASE
)

NUM_FREQ_RE = re.compile(r"^\s*Number of frequencies\s*:\s*(\d+)\s*$", re.IGNORECASE)


# frequency: 0.0000 (au) kind of lines (for gradient blocks etc)
FREQ_LINE_RE = re.compile(
    r"^\s*frequency\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$",
    re.IGNORECASE,
)

POLAR_FREQ_TITLE_RE = re.compile(
    r"^\s*\++\s*Frequency dependent polarizabilities\s*\++\s*$",
    re.IGNORECASE,
)

POLAR_TENSOR_HDR_RE = re.compile(
    r"^\s*Polarizability tensor for frequency\s+"
    r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*au\s*$"
)

ISO_POLAR_RE = re.compile(
    r"^\s*Isotropic polarizability:\s*" r"([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)

POLAR_CART_TITLE_RE = re.compile(
    r"^\s*Polarizability tensor Cartesian gradient\)?\s*$", re.IGNORECASE
)
POLAR_NORMAL_TITLE_RE = re.compile(
    r"^\s*Polarizability tensor normal coord\. gradient\s*$", re.IGNORECASE
)

# generic float: handles 23.0, -1.2E-09, 3.14e+02, etc.
FLOAT_RE = re.compile(r"[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?")

# Vibrational table
VIB_TITLE_RE = re.compile(
    r"^\s*Vibrational Frequencies and IR Intensities\s*$", re.IGNORECASE
)
MODE_ROW_RE = re.compile(
    r"""^\s*
        (\d+)                     # mode
        \s+([A-Za-z0-9]+)         # irrep
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # freq cm^-1
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # freq hartree
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # IR km/mol
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # (D/A)^2/amu
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Raman table
RAMAN_TITLE_RE = re.compile(
    r"^\s*Raman related properties for freq\.\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*au\b",
    re.IGNORECASE,
)
RAMAN_ROW_RE = re.compile(
    r"""^\s*
        (\d+)
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # freq cm^-1
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # alpha^2
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # beta(a)^2
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # Pol.Int.
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # Depol.Int.
        \s+([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)   # Dep. Ratio
        \s*$""",
    re.IGNORECASE | re.VERBOSE,
)

# Final HF energy
HF_ENERGY_RE = re.compile(r"^@\s*Final HF energy:\s*(-?\d+\.\d+)\s*$")

# --- add near your other regexes ---
ORBITAL_SECTION_HDR_RE = re.compile(
    r"^\s*\*{3}\s*SCF orbital energy analysis\s*\*{3}\s*$", re.IGNORECASE
)
NUM_ELECTRONS_RE = re.compile(r"^\s*Number of electrons\s*:\s*(\d+)\s*$", re.IGNORECASE)
ORBITAL_OCC_RE = re.compile(r"^\s*Orbital occupations\s*:\s*(\d+)\s*$", re.IGNORECASE)
SYMM_ORB_HDR_RE = re.compile(
    r"^\s*Sym\s+Hartree-Fock orbital energies\s*$", re.IGNORECASE
)

SYM_BLOCK_START_RE = re.compile(
    r"""^\s*
        (\d+)                # symmetry index
        \s+([A-Za-z0-9]+)    # irrep label
        \s+(.+?)             # rest of line = energies
        \s*$""",
    re.VERBOSE,
)

LUMO_RE = re.compile(
    r"^\s*E\(LUMO\)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE
)
HOMO_RE = re.compile(
    r"^\s*-?\s*E\(HOMO\)\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE
)
GAP_RE = re.compile(
    r"^\s*gap\s*:\s*([+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)", re.IGNORECASE
)


# ---------- Small helpers ----------


def _find_line_matching(
    lines: Sequence[str],
    pattern: Union[str, re.Pattern],
    start: int = 0,
) -> Optional[int]:
    pat = re.compile(pattern) if isinstance(pattern, str) else pattern
    for i in range(start, len(lines)):
        if pat.search(lines[i]):
            return i
    return None


def _parse_col_numbers(line: str) -> List[int]:
    return list(map(int, re.findall(r"Column\s+(\d+)", line)))


def parse_mass_weighted_hessian_eigenvalues(
    lines: Sequence[str],
    natoms: int,
    start_index: int = 0,
) -> Tuple[np.ndarray, int]:
    """
    Parse:
        Eigenvalues of mass-weighted Hessian
        ------------------------------------
           Column 1  Column 2 ...
        1  3.49E-39  -1.96E-21  ...

    Returns
    -------
    eigvals : (3N,) np.ndarray
    next_idx : int
    """
    if natoms is None:
        raise ValueError(
            "natoms is required to parse mass-weighted Hessian eigenvalues"
        )

    nrows = 1
    ncols = 3 * natoms
    mat, next_idx = parse_dalton_matrix(
        lines=lines,
        header_pattern=MWH_EIGEN_HDR_RE,
        nrows=nrows,
        ncols=ncols,
        start_index=start_index,
    )
    eigvals = mat[0, :].copy()
    return eigvals, next_idx


# ---------- Polarizability tensor (3x3 per frequency) ----------


def parse_frequency_polarizability_tensors(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[Dict[float, Dict[str, Any]], int]:
    """
    Parse blocks like:

      Polarizability tensor for frequency     0.000000 au
      ----------------------------------------------------
                            Ex             Ey             Ez

                Ex    23.00937     -1.1685992E-09 -1.0240027E-08
                Ey  -1.1705532E-09   24.82077     -4.8148781E-08
                Ez   4.3458050E-09  2.2710151E-08   36.96431

      Isotropic polarizability:   28.26482

    Returns
    -------
    pol_by_freq : dict
        { freq_au : {"tensor": 3x3 np.ndarray, "iso": float} }
    next_idx : int
        Line index where scanning ended (after the last parsed block).
    """
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

        # Skip dashed line + blanks until we see the "Ex Ey Ez" column header
        while i < nlines and "Ex" not in lines[i]:
            i += 1
        if i >= nlines:
            break  # incomplete block at end of file

        # lines[i] is column header; advance to first data row
        i += 1
        while i < nlines and not lines[i].strip():
            i += 1

        tensor = np.zeros((3, 3), dtype=float)
        for row in range(3):
            if i >= nlines:
                raise ValueError(
                    "Unexpected end of file while reading polarizability tensor rows."
                )
            line = lines[i]
            nums = FLOAT_RE.findall(line)
            if len(nums) < 3:
                raise ValueError(
                    f"Could not parse three floats from tensor row: {line!r}"
                )
            vals = [float(x) for x in nums[-3:]]
            tensor[row, :] = vals
            i += 1

        # Find isotropic polarizability
        iso_val: Optional[float] = None
        while i < nlines:
            line = lines[i]
            m_iso = ISO_POLAR_RE.match(line)
            if m_iso:
                iso_val = float(m_iso.group(1))
                i += 1
                break
            # If we hit another header, bail out to outer loop
            if POLAR_TENSOR_HDR_RE.match(line):
                break
            i += 1

        pol_by_freq[freq_au] = {"tensor": tensor, "iso": iso_val}

    return pol_by_freq, i


def _parse_polarizability_section(
    lines: Sequence[str],
    start_index: int = 0,
    nfreqs: Optional[int] = None,
) -> Tuple[np.ndarray, List[float], int]:
    """
    Wrapper to parse the 'Frequency dependent polarizabilities' section
    and return an array of 3x3 tensors ordered by frequency.
    """
    # Find the last "++++++++++++++++++ Frequency dependent polarizabilities ++++++++++++++++++"
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
        raise ValueError("Found polarizability section header but no tensors.")

    freqs_sorted = sorted(pol_by_freq.keys())
    if nfreqs is not None:
        freqs_sorted = freqs_sorted[:nfreqs]

    tensors = np.array([pol_by_freq[f]["tensor"] for f in freqs_sorted])
    return tensors, freqs_sorted, next_idx


# ---------- Generic Dalton matrix parser ----------


def parse_dalton_matrix(
    lines: Sequence[str],
    header_pattern: Union[str, re.Pattern],
    nrows: int,
    ncols: int,
    start_index: int = 0,
) -> Tuple[np.ndarray, int]:
    """Generic parser for Dalton matrices printed in ≤6-column blocks."""
    HEADER_RE = (
        re.compile(header_pattern)
        if isinstance(header_pattern, str)
        else header_pattern
    )

    hdr_i = _find_line_matching(lines, HEADER_RE, start=start_index)
    if hdr_i is None:
        raise ValueError("Matrix header not found.")
    i = hdr_i + 1

    # Seek first "Column ..." header
    while i < len(lines) and not COL_HEADER_RE.match(lines[i]):
        if END_MATRIX_RE.match(lines[i]):
            raise ValueError("Reached end-of-matrix marker before any column header.")
        i += 1
    if i >= len(lines):
        raise ValueError("Column header not found after matrix header.")

    mat = np.zeros((nrows, ncols), dtype=float)

    # Iterate over column blocks
    while i < len(lines):
        if END_MATRIX_RE.match(lines[i]):
            i += 1
            break

        if not COL_HEADER_RE.match(lines[i]):
            if lines[i].strip() == "":
                i += 1
                continue
            if HEADER_RE.search(lines[i]):
                break
            i += 1
            continue

        col_nums = _parse_col_numbers(lines[i])
        col_nums = [c for c in col_nums if 1 <= c <= ncols]
        block_width = len(col_nums)
        i += 1  # first row line of this block

        rows_read = 0
        while rows_read < nrows and i < len(lines):
            line = lines[i].rstrip("\n")
            if END_MATRIX_RE.match(line):
                i += 1
                break
            if not line.strip():
                i += 1
                continue

            parts = line.split()
            if not parts or not parts[0].isdigit():
                break

            row_idx = int(parts[0])
            if not (1 <= row_idx <= nrows):
                raise ValueError(f"Row index out of bounds at line {i}: {line!r}")

            values = parts[1:]
            if len(values) < block_width:
                raise ValueError(
                    f"Expected {block_width} values for columns {col_nums}, got {len(values)} at line {i}: {line!r}"
                )

            for k, col in enumerate(col_nums):
                mat[row_idx - 1, col - 1] = float(values[k])

            rows_read += 1
            i += 1

        # skip blank lines between blocks
        while i < len(lines) and lines[i].strip() == "":
            i += 1
        if i < len(lines) and END_MATRIX_RE.match(lines[i]):
            i += 1
            break
        if i >= len(lines) or not COL_HEADER_RE.match(lines[i]):
            break

    return mat, i


# ---------- Geometry ----------


def parse_last_molecular_geometry(
    lines: Sequence[str],
    *,
    header_skip: int = 3,
    return_raw: bool = False,
) -> qcel.models.Molecule | Tuple[qcel.models.Molecule, List[str], List[List[float]]]:
    """Parse the LAST 'Molecular geometry (...)' block, returns Molecule in Å."""
    last_header_idx: Optional[int] = None
    last_unit: Optional[str] = None
    for i, line in enumerate(lines):
        m = MOLECULAR_GEOMETRY_RE.match(line)
        if m:
            last_header_idx = i

    if last_header_idx is None:
        raise ValueError("No 'Molecular geometry (...)' header found.")

    unit = "au"  # default
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
        print(parts)
        if len(parts) == 5:
            start_float_index = 2
        else:
            start_float_index = 1
        x, y, z = (
            float(parts[start_float_index]),
            float(parts[start_float_index + 1]),
            float(parts[start_float_index + 2]),
        )
        atoms.append(label)
        coords.append([x, y, z])

    if not atoms:
        raise ValueError("Found geometry header but no atom lines followed.")

    if unit == "au":
        bohr_to_ang = qcel.constants.bohr2angstroms
        coords = [[c * bohr_to_ang for c in vec] for vec in coords]

    mol = qcel.models.Molecule(symbols=atoms, geometry=coords)
    return (mol, atoms, coords) if return_raw else mol


# ---------- Optimized Geometry ----------
def parse_optimized_geometry(
    lines: Sequence[str],
    *,
    header_skip: int = 3,
    return_raw: bool = False,
) -> qcel.models.Molecule | Tuple[qcel.models.Molecule, List[str], List[List[float]]]:
    """Parse the LAST 'Optimized geometry (...)' block, returns Molecule in Å."""
    last_header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        m = OPTIMIZED_GEOMETRY_RE.match(line)
        if m:
            last_header_idx = i

    if last_header_idx is None:
        raise ValueError("No 'Optimized geometry (...)' header found.")

    unit = "au"  # default
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
        x, y, z = (float(parts[1]), float(parts[2]), float(parts[3]))
        atoms.append(label)
        coords.append([x, y, z])

    if not atoms:
        raise ValueError("Found geometry header but no atom lines followed.")

    if unit == "au":
        bohr_to_ang = qcel.constants.bohr2angstroms
        coords = [[c * bohr_to_ang for c in vec] for vec in coords]

    mol = qcel.models.Molecule(symbols=atoms, geometry=coords)
    return (mol, atoms, coords) if return_raw else mol


# ---------- Frequencies & polarizability gradients ----------


def parse_number_of_frequencies(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[int, int]:
    idx = _find_line_matching(lines, NUM_FREQ_RE, start=start_index)
    if idx is None:
        raise ValueError("Could not find 'Number of frequencies : ...' line.")

    m = NUM_FREQ_RE.match(lines[idx])
    if not m:
        raise ValueError("Malformed 'Number of frequencies' line.")
    nfreq = int(m.group(1))
    return nfreq, idx + 1


def parse_polarizability_gradients_all(
    lines,
    nrows: int,
    ncols: int,
    number_of_frequencies: int = 1,
    title_re=POLAR_CART_TITLE_RE,
    start_index: int = 0,
) -> Tuple[Dict[float, np.ndarray], int]:
    """
    Parse 'Polarizability tensor ... gradient' blocks at multiple frequencies.

    Parameters
    ----------
    nrows, ncols : int
        Dimensions of the printed matrix, e.g.
        - Cartesian gradient: 9 x (3 * natoms)
        - Normal-coordinate gradient: 9 x Nmodes
    ...
    """
    title_i = _find_line_matching(lines, title_re, start=start_index)
    if title_i is None:
        raise ValueError("Could not find polarizability gradient title.")

    results: Dict[float, np.ndarray] = {}
    freq_vals: List[float] = []
    block_index = title_i + 1

    for _ in range(number_of_frequencies):
        freq_i = _find_line_matching(lines, FREQ_LINE_RE, start=block_index)
        if freq_i is None:
            # No more blocks
            break

        m = FREQ_LINE_RE.match(lines[freq_i])
        if not m:
            raise ValueError(
                f"Malformed 'frequency: ...' line at {freq_i}: {lines[freq_i]!r}"
            )

        freq_val = float(m.group(1))
        freq_vals.append(freq_val)

        mat, next_idx = parse_dalton_matrix(
            lines=lines,
            header_pattern=FREQ_LINE_RE,
            nrows=nrows,
            ncols=ncols,
            start_index=freq_i,
        )
        block_index = next_idx
        results[freq_val] = mat

    return results, block_index


# ---------- Vibrational table ----------


def parse_vibrational_frequencies_table(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[List[Dict[str, Any]], int]:
    """Parse vibrational modes table; returns (rows, next_idx)."""
    i = start_index
    while i < len(lines) and not VIB_TITLE_RE.match(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("Vibrational frequencies section not found.")

    # Advance to first data row
    i += 1
    while i < len(lines) and not MODE_ROW_RE.match(lines[i]):
        i += 1
    if i >= len(lines):
        raise ValueError("No mode rows found after vibrational header.")

    rows: List[Dict[str, Any]] = []
    while i < len(lines):
        line = lines[i]
        if BLANK_RE.match(line):
            i += 1
            break
        if SEP_RE.match(line):
            i += 1
            continue

        m = MODE_ROW_RE.match(line)
        if not m:
            break

        rows.append(
            dict(
                mode=int(m.group(1)),
                irrep=m.group(2),
                freq_cm1=float(m.group(3)),
                freq_hartree=float(m.group(4)),
                ir_km_per_mol=float(m.group(5)),
                dipole2_per_amu=float(m.group(6)),
            )
        )
        i += 1

    return rows, i


# ---------- Raman tables ----------


def parse_all_raman_tables(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[Dict[float, List[Dict[str, Any]]], int]:
    """Parse all Raman property tables across file; returns ({freq_au: rows}, next_idx)."""
    results: Dict[float, List[Dict[str, Any]]] = {}
    idx = start_index
    VIB_ROI_RE = re.compile(
        r"^\s* \*{3}\s* Output from VIBRO1 \*{3}\s*$", re.IGNORECASE
    )
    # find the start of the last VIBRO1 output
    last_vibro1_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if VIB_ROI_RE.match(line):
            last_vibro1_idx = i
    if last_vibro1_idx is not None:
        idx = last_vibro1_idx

    while True:
        j = _find_line_matching(lines, RAMAN_TITLE_RE, start=idx)
        if j is None:
            break
        m = RAMAN_TITLE_RE.match(lines[j])
        if not m:
            break
        freq_au = float(m.group(1))

        j += 1
        while j < len(lines) and not RAMAN_ROW_RE.match(lines[j]):
            j += 1
        if j >= len(lines):
            break

        rows: List[Dict[str, Any]] = []
        while j < len(lines) and not BLANK_RE.match(lines[j]):
            if SEP_RE.match(lines[j]):
                j += 1
                continue
            rm = RAMAN_ROW_RE.match(lines[j])
            if not rm:
                break
            rows.append(
                dict(
                    mode=int(rm.group(1)),
                    freq_cm1=float(rm.group(2)),
                    alpha2=float(rm.group(3)),
                    beta2=float(rm.group(4)),
                    pol_int=float(rm.group(5)),
                    depol_int=float(rm.group(6)),
                    depol_ratio=float(rm.group(7)),
                )
            )
            j += 1

        results[freq_au] = rows
        idx = j + 1

    return results, idx


# ---------- HF energy ----------


def parse_hf_energy(lines: Sequence[str]) -> float:
    for line in lines:
        m = HF_ENERGY_RE.match(line.strip())
        if m:
            return float(m.group(1))
    raise ValueError("Final HF energy line not found.")


def parse_scf_orbital_energies(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[Dict[str, Any], int]:
    """
    Parse an 'SCF orbital energy analysis' block.

    Returns
    -------
    info : dict
        {
          "nelec": int or None,
          "nocc": int or None,             # occupations per irrep (Dalton's number)
          "energies_by_sym": {             # per symmetry
              "1 A": np.ndarray([...]),
              ...
          },
          "homo": float or None,
          "lumo": float or None,
          "gap": float or None,
        }
    next_idx : int
        Line index after the block (after gap line or last line scanned).
    """
    nlines = len(lines)

    # 1) Find the section header
    hdr_i = _find_line_matching(lines, ORBITAL_SECTION_HDR_RE, start=start_index)
    if hdr_i is None:
        raise ValueError("*** SCF orbital energy analysis *** section not found.")

    i = hdr_i + 1
    nelec = None
    nocc = None

    # 2) Scan down to Sym ... header, picking up metadata
    while i < nlines:
        line = lines[i]

        m_ne = NUM_ELECTRONS_RE.match(line)
        if m_ne:
            nelec = int(m_ne.group(1))

        m_occ = ORBITAL_OCC_RE.match(line)
        if m_occ:
            nocc = int(m_occ.group(1))

        if SYMM_ORB_HDR_RE.match(line):
            break

        i += 1

    if i >= nlines:
        raise ValueError("'Sym Hartree-Fock orbital energies' header not found.")

    # move to first line after "Sym ..." header
    i += 1

    # 3) Read symmetry blocks with orbital energies
    energies_by_sym: Dict[str, list[float]] = {}
    current_sym_key: str | None = None

    # Skip leading blanks
    while i < nlines and BLANK_RE.match(lines[i]):
        i += 1

    while i < nlines:
        line = lines[i]

        # Stop if we hit the LUMO/HOMO/gap summary
        if LUMO_RE.match(line) or HOMO_RE.match(line) or GAP_RE.match(line):
            break

        # Blank lines can separate symmetry blocks; just skip them
        if BLANK_RE.match(line):
            i += 1
            continue

        # New symmetry block?
        m_start = SYM_BLOCK_START_RE.match(line)
        if m_start:
            sym_idx = int(m_start.group(1))
            irrep = m_start.group(2)
            rest = m_start.group(3)

            key = f"{sym_idx} {irrep}"
            current_sym_key = key
            if key not in energies_by_sym:
                energies_by_sym[key] = []

            nums = FLOAT_RE.findall(rest)
            energies_by_sym[key].extend(float(x) for x in nums)
            i += 1
            continue

        # Continuation lines for current symmetry: just more energies
        if current_sym_key is not None:
            nums = FLOAT_RE.findall(line)
            if nums:
                energies_by_sym[current_sym_key].extend(float(x) for x in nums)
                i += 1
                continue

        # Anything else → probably out of the orbital table
        if line.strip().startswith("E(") or line.strip().startswith("gap"):
            break

        i += 1

    # 4) Parse HOMO / LUMO / gap summary
    lumo = homo = gap = None

    while i < nlines:
        line = lines[i]
        mL = LUMO_RE.match(line)
        if mL:
            lumo = float(mL.group(1))

        mH = HOMO_RE.match(line)
        if mH:
            homo = float(mH.group(1))

        mG = GAP_RE.match(line)
        if mG:
            gap = float(mG.group(1))

        # Stop once we got all three (or reach a separator)
        if gap is not None and lumo is not None and homo is not None:
            i += 1
            break

        if SEP_RE.match(line):
            i += 1
            break

        i += 1

    # 5) Convert lists → np arrays
    energies_by_sym_np: Dict[str, Any] = {
        key: np.array(vals, dtype=float) for key, vals in energies_by_sym.items()
    }

    info = {
        "nelec": nelec,
        "nocc": nocc,
        "energies_by_sym": energies_by_sym_np,
        "homo": homo,
        "lumo": lumo,
        "gap": gap,
    }

    return info, i


# ---------- High-level parser class ----------


class DaltonParser:
    """
    High-level Dalton output parser.

    Attributes (filled after parse_* calls or parse_all()):
      lines: list[str]
      molecule: qcel.models.Molecule
      atoms: list[str]
      coords: list[list[float]] (Å)
      natoms: int
      hessian: np.ndarray (3N x 3N)
      normal_coordinates: np.ndarray (3N x 3N)
      vib_rows: list[dict]  (vibrational modes table)
      nfreq: int
      polarizability: np.ndarray          # shape (nfreq, 3, 3)
      polar_cart_grad: dict[float, np.ndarray]     # freq (au) -> (3N x ncols)
      polar_normal_grad: dict[float, np.ndarray]   # freq (au) -> (Nmodes x ncols)
      raman_by_freq: dict[float, list[dict]]
      hf_energy: float
    """

    def __init__(self, src: Union[str, Path, Sequence[str]]):
        if isinstance(src, (str, Path)):
            with open(src, "r") as f:
                self.lines = f.readlines()
        else:
            self.lines = list(src)

        self.molecule: Optional[qcel.models.Molecule] = None
        self.atoms: Optional[List[str]] = None
        self.coords: Optional[List[List[float]]] = None
        self.natoms: Optional[int] = None

        self.hessian: Optional[np.ndarray] = None
        self.normal_coordinates: Optional[np.ndarray] = None
        self.mwh_eigvals: Optional[List[float]] = None  # eigenvalues in mw units

        self.vib_rows: Optional[List[Dict[str, Any]]] = None
        self.nfreq: Optional[int] = None

        self.polarizability: Optional[np.ndarray] = None
        self.polar_cart_grad: Optional[Dict[float, np.ndarray]] = None
        self.polar_normal_grad: Optional[Dict[float, np.ndarray]] = None

        self.raman_by_freq: Optional[Dict[float, List[Dict[str, Any]]]] = None
        self.hf_energy: Optional[float] = None
        self.orbital_info: Optional[Dict[str, Any]] = None

    # ---------- individual parsers ----------

    def parse_geometry(self) -> qcel.models.Molecule:
        mol, atoms, coords = parse_last_molecular_geometry(self.lines, return_raw=True)
        self.molecule, self.atoms, self.coords = mol, atoms, coords
        self.natoms = len(atoms)
        return mol

    def parse_optimized_geometry(self) -> qcel.models.Molecule:
        return self.parse_geometry()

    def parse_hessian(self) -> np.ndarray:
        if self.natoms is None:
            self.parse_geometry()
        H, _ = parse_dalton_matrix(
            self.lines, HESSIAN_HDR_RE, 3 * self.natoms, 3 * self.natoms
        )
        self.hessian = H
        return H

    def parse_normal_coordinates(self) -> np.ndarray:
        if self.natoms is None:
            self.parse_geometry()
        Q, _ = parse_dalton_matrix(
            self.lines, NORMAL_COORDS_HDR_RE, 3 * self.natoms, 3 * self.natoms
        )
        self.normal_coordinates = Q
        return Q

    def parse_mass_weighted_eigenvalues(self) -> np.ndarray:
        if self.natoms is None:
            self.parse_geometry()
        eig, _ = parse_mass_weighted_hessian_eigenvalues(
            self.lines, self.natoms, start_index=0
        )
        self.mwh_eigvals = eig
        return eig

    def significant_mode_indices(
        self,
        *,
        tol: float = 1e-8,
        require_positive: bool = True,
    ) -> np.ndarray:
        """
        Return indices (0..3N-1) of normal modes considered usable for comparison.

        tol : float
            Magnitude threshold to drop near-zero (translation/rotation) modes.
        require_positive : bool
            If True, only keep strictly positive eigenvalues (no imaginary modes).

        Notes
        -----
        - The eigenvalues here are those of the mass-weighted Hessian. Their sign
          indicates stable (+) vs imaginary (–) curvature.
        """
        if self.mwh_eigvals is None:
            self.parse_mass_weighted_eigenvalues()
        eig = self.mwh_eigvals
        mask = np.abs(eig) > tol
        if require_positive:
            mask &= eig > 0.0
        return np.where(mask)[0]

    def parse_vibrations(self) -> List[Dict[str, Any]]:
        rows, _ = parse_vibrational_frequencies_table(self.lines)
        self.vib_rows = rows
        self.vibrational_frequencies = [row["freq_cm1"] for row in rows]
        return rows

    def parse_num_frequencies(self) -> int:
        n, _ = parse_number_of_frequencies(self.lines, start_index=0)
        self.nfreq = n
        return n

    def parse_polarizability(self) -> np.ndarray:
        if self.nfreq is None:
            self.parse_num_frequencies()
        tensors, freqs, _ = _parse_polarizability_section(
            self.lines,
            start_index=0,
            nfreqs=self.nfreq,
        )
        self.polarizability = tensors
        # if you want frequencies too, you could stash them on self
        # self.polar_freqs = freqs
        return tensors

    def parse_polar_cartesian_gradients(self) -> Dict[float, np.ndarray]:
        if self.natoms is None:
            self.parse_geometry()
        if self.nfreq is None:
            self.parse_num_frequencies()

        # 9 rows (tensor components), 3N columns (Cartesian coordinates)
        nrows = 9
        ncols = 3 * self.natoms

        mats, _ = parse_polarizability_gradients_all(
            self.lines,
            nrows=nrows,
            ncols=ncols,
            number_of_frequencies=self.nfreq,
            title_re=POLAR_CART_TITLE_RE,
            start_index=0,
        )
        self.polar_cart_grad = mats
        return mats

    def parse_polar_normal_gradients(self) -> Dict[float, np.ndarray]:
        if self.vib_rows is None:
            self.parse_vibrations()
        if self.nfreq is None:
            self.parse_num_frequencies()

        Nmodes = len(self.vib_rows)

        # 9 rows (tensor components), Nmodes columns (normal modes)
        nrows = 9
        ncols = Nmodes

        mats, _ = parse_polarizability_gradients_all(
            self.lines,
            nrows=nrows,
            ncols=ncols,
            number_of_frequencies=self.nfreq,
            title_re=POLAR_NORMAL_TITLE_RE,
            start_index=0,
        )
        self.polar_normal_grad = mats
        return mats

    def parse_raman_tables(self) -> Dict[float, List[Dict[str, Any]]]:
        res, _ = parse_all_raman_tables(self.lines)
        self.raman_by_freq = res
        return res

    def parse_final_hf_energy(self) -> float:
        e = parse_hf_energy(self.lines)
        self.hf_energy = e
        return e

    def parse_orbitals(self) -> Dict[str, Any]:
        info, _ = parse_scf_orbital_energies(self.lines)
        self.orbital_info = info
        return info

    # ---------- convenience ----------

    def parse_all(self) -> "DaltonParser":
        """Run all parsers with sensible defaults."""
        self.parse_geometry()
        self.parse_hessian()
        self.parse_normal_coordinates()
        self.parse_vibrations()
        self.parse_num_frequencies()
        self.parse_polarizability()
        self.parse_polar_cartesian_gradients()
        self.parse_polar_normal_gradients()
        self.parse_raman_tables()
        self.parse_final_hf_energy()
        self.parse_orbitals()
        return self

    def summary(self) -> Dict[str, Any]:
        """Lightweight dictionary for downstream comparison/analytics."""
        return {
            "natoms": self.natoms,
            "hf_energy": self.hf_energy,
            "nmodes": None if self.vib_rows is None else len(self.vib_rows),
            "nfreq": self.nfreq,
            "hessian_shape": (
                None if self.hessian is None else tuple(self.hessian.shape)
            ),
            "normal_coords_shape": (
                None
                if self.normal_coordinates is None
                else tuple(self.normal_coordinates.shape)
            ),
            "polar_cart_freqs": (
                None
                if self.polar_cart_grad is None
                else sorted(self.polar_cart_grad.keys())
            ),
            "polar_normal_freqs": (
                None
                if self.polar_normal_grad is None
                else sorted(self.polar_normal_grad.keys())
            ),
            "raman_freqs": (
                None
                if self.raman_by_freq is None
                else sorted(self.raman_by_freq.keys())
            ),
        }


