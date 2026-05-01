from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple

import numpy as np
import qcelemental as qcel

from gecko.core.model import Calculation
from gecko.molecule.canonical import canonicalize_atom_order
from gecko.plugins.dalton.legacy.dalton import DaltonParser as LegacyDaltonParser

try:
    from daltonproject.dalton.output_parser import OutputParser as _DaltonProjectOutputParser
    _DALTONPROJECT_AVAILABLE = True
except ImportError:
    _DALTONPROJECT_AVAILABLE = False


_BETA_LINE_PREFIX = "@ B-freq"

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
MOL_BLOCK_HDR_RE = re.compile(r"^\s*Content of the \.mol file\s*$", re.IGNORECASE)
EXCITATION_SECTION_RE = re.compile(
    r"^\s*@?\s*(?P<spin>\w+)\s+electronic excitation energies\s*$",
    re.IGNORECASE,
)
EXCITATION_ROW_RE = re.compile(
    r"^\s*(?P<sym>\d+)\s+(?P<mode>\d+)\s+"
    r"(?P<omega_au>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s+"
    r"(?P<omega_ev>[+-]?\d+(?:\.\d+)?(?:[Ee][+-]?\d+)?)\s*$"
)


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()

def read_dalton_mol(path: Path) -> qcel.models.Molecule:
    """Read a Dalton-format `.mol` file into a qcelemental Molecule."""
    mol_path = Path(path).expanduser().resolve()
    lines = _read_lines(mol_path)
    return parse_molfile_geometry(lines)

def infer_basis_from_dalton_mol(path: Path) -> str | None:
    """Infer the Dalton basis name from a Dalton `.mol` file."""
    mol_path = Path(path).expanduser().resolve()
    lines = _read_lines(mol_path)
    for i, line in enumerate(lines):
        if line.strip().upper() == "BASIS":
            for j in range(i + 1, len(lines)):
                if lines[j].strip():
                    return lines[j].strip()
            return None
    return None


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
    atoms: list[str] = []
    coords: list[list[float]] = []

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
    atoms, coords = canonicalize_atom_order(atoms, coords, decimals=10)

    return qcel.models.Molecule(symbols=atoms, geometry=coords)


def parse_molfile_geometry(lines: Sequence[str]) -> qcel.models.Molecule:
    units = "angstrom"
    atoms: list[str] = []
    coords: list[list[float]] = []

    for line in lines:
        if "angstrom" in line.lower():
            units = "angstrom"
        elif "bohr" in line.lower():
            units = "bohr"

    for line in lines:
        if not line.strip():
            continue
        if line.strip().startswith("Charge=") or line.strip().startswith("Charge:"):
            continue
        if line.strip().startswith("Atomtype="):
            continue
        if line.strip().upper() in {"BASIS"}:
            continue
        if line.strip().upper().startswith("END"):
            break

        parts = line.split()
        if len(parts) >= 4 and parts[0][0].isalpha():
            label = parts[0].split("_", 1)[0]
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            atoms.append(label)
            coords.append([x, y, z])

    if not atoms:
        raise ValueError("No atom coordinates found in mol file")

    if units == "bohr":
        bohr_to_ang = qcel.constants.bohr2angstroms
        coords = [[c * bohr_to_ang for c in vec] for vec in coords]
    atoms, coords = canonicalize_atom_order(atoms, coords, decimals=10)
    return qcel.models.Molecule(symbols=atoms, geometry=coords)


def parse_mol_block_from_output(lines: Sequence[str]) -> tuple[qcel.models.Molecule, Optional[str]]:
    header_idx: Optional[int] = None
    for i, line in enumerate(lines):
        if MOL_BLOCK_HDR_RE.match(line):
            header_idx = i
            break
    if header_idx is None:
        raise ValueError("No 'Content of the .mol file' block found.")

    block: list[str] = []
    started = False
    found_atoms = False
    sep_re = re.compile(r"^\s*-{3,}\s*$")

    for j in range(header_idx + 1, len(lines)):
        line = lines[j]
        if sep_re.match(line):
            if started:
                continue
            continue

        if not line.strip():
            if found_atoms:
                break
            if not started:
                continue
            continue

        started = True
        block.append(line)
        if line.strip().startswith("Charge="):
            continue
        if line.strip().startswith("Atomtype="):
            continue
        parts = line.split()
        if len(parts) >= 4 and parts[0][0].isalpha():
            found_atoms = True

    if not block:
        raise ValueError("Empty .mol block in output.")

    basis_name: Optional[str] = None
    for i, line in enumerate(block):
        if line.strip().upper() == "BASIS":
            for j in range(i + 1, len(block)):
                if block[j].strip():
                    basis_name = block[j].strip()
                    break
            break

    return parse_molfile_geometry(block), basis_name


def parse_frequency_polarizability_tensors(
    lines: Sequence[str],
    start_index: int = 0,
) -> Tuple[dict[float, dict[str, Any]], int]:
    i = start_index
    pol_by_freq: dict[float, dict[str, Any]] = {}
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
) -> Tuple[np.ndarray, list[float], int]:
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


def _is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and set(stripped) <= {"-", "="}


def parse_electronic_excitations(lines: Sequence[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    nlines = len(lines)
    i = 0

    while i < nlines:
        section_match = EXCITATION_SECTION_RE.match(lines[i])
        if not section_match:
            i += 1
            continue

        spin = section_match.group("spin").lower()
        i += 1
        saw_data = False

        while i < nlines:
            row_match = EXCITATION_ROW_RE.match(lines[i])
            if row_match:
                saw_data = True
                rows.append(
                    {
                        "spin": spin,
                        "symmetry_excited_state": int(row_match.group("sym")),
                        "mode": int(row_match.group("mode")),
                        "omega_au": float(row_match.group("omega_au")),
                        "omega_ev": float(row_match.group("omega_ev")),
                    }
                )
                i += 1
                continue

            stripped = lines[i].strip()
            if saw_data and not stripped:
                break
            if saw_data and stripped and not _is_separator_line(lines[i]):
                break
            i += 1

    return rows


class DaltonParser:
    def __init__(self, src: str | Path | Sequence[str]):
        if isinstance(src, (str, Path)):
            self.lines = _read_lines(Path(src))
        else:
            self.lines = list(src)

        self.molecule: Optional[qcel.models.Molecule] = None
        self.hf_energy: Optional[float] = None
        self.mol_block_basis: Optional[str] = None

    def parse_geometry(self) -> qcel.models.Molecule:
        try:
            mol, basis = parse_mol_block_from_output(self.lines)
            self.molecule = mol
            self.mol_block_basis = basis
        except Exception:
            self.molecule = parse_last_molecular_geometry(self.lines)
        return self.molecule

    def parse_final_hf_energy(self) -> float:
        self.hf_energy = parse_hf_energy(self.lines)
        return self.hf_energy


def _infer_basis_from_content(lines: list[str]) -> str | None:
    patterns = [
        re.compile(r"^\s*Basis\s+set\s*:\s*(.+)$", re.IGNORECASE),
        re.compile(r"^\s*Basis\s*:\s*(.+)$", re.IGNORECASE),
    ]
    for line in lines[:400]:
        for pat in patterns:
            m = pat.match(line)
            if m:
                return m.group(1).strip()
    return None


def split_label_basis_from_outname(name: str) -> tuple[str | None, str | None]:
    stem = name
    if stem.endswith(".out"):
        stem = stem[:-4]

    basis_patterns = [
        re.compile(r"^(?P<label>.+?)-(?P<basis>(?:d-)?aug-cc-.+)$"),
        re.compile(r"^(?P<label>.+?)-(?P<basis>cc-.+)$"),
    ]
    for pat in basis_patterns:
        m = pat.match(stem)
        if m:
            return m.group("label"), m.group("basis")

    if "-" in stem:
        label, basis = stem.rsplit("-", 1)
        return label, basis
    return stem, None


def _alpha_tensor_to_data(tensors: np.ndarray, freqs: list[float]) -> dict[str, Any]:
    components = ["xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz"]
    values: list[list[float]] = []
    for t in tensors:
        values.append(
            [
                float(t[0, 0]),
                float(t[0, 1]),
                float(t[0, 2]),
                float(t[1, 0]),
                float(t[1, 1]),
                float(t[1, 2]),
                float(t[2, 0]),
                float(t[2, 1]),
                float(t[2, 2]),
            ]
        )
    return {
        "omega": np.asarray(freqs, dtype=float),
        "components": components,
        "values": np.asarray(values, dtype=float),
        "shape": ("freq", "component"),
    }


def _parse_beta_from_quad_lines(lines: list[str]) -> dict[str, Any] | None:
    start_idx = None
    for i, line in enumerate(lines):
        if "Results from quadratic response calculation" in line:
            start_idx = i
            break
    if start_idx is None:
        return None

    pattern = re.compile(
        r"@\s*B-freq\s*=\s*([\d\.Ee+-]+)\s+"
        r"C-freq\s*=\s*([\d\.Ee+-]+)\s+"
        r"beta\((X|Y|Z);(X|Y|Z),(X|Y|Z)\)\s*=\s*(.+)$"
    )

    freq_map: dict[tuple[float, float, float], dict[str, float]] = {}
    alias_map: dict[tuple[float, float, float], dict[str, str]] = {}
    alias_pattern = re.compile(r"beta\((X|Y|Z)(?:;|,)?(X|Y|Z),(X|Y|Z)\)")

    for line in lines[start_idx:]:
        if not line.startswith(_BETA_LINE_PREFIX):
            continue
        match = pattern.search(line)
        if not match:
            continue
        b_freq, c_freq, a1, a2, a3, value_raw = match.groups()
        value = None
        try:
            value = float(str(value_raw).split()[0])
        except ValueError:
            value = None

        b = float(b_freq)
        c = float(c_freq)
        a = -(b + c)
        ijk = f"{a1}{a2}{a3}".lower()
        key = (a, b, c)
        if value is not None:
            freq_map.setdefault(key, {})[ijk] = value
        else:
            alias_match = alias_pattern.search(str(value_raw))
            if alias_match:
                b1, b2, b3 = alias_match.groups()
                alias_comp = f"{b1}{b2}{b3}".lower()
                alias_map.setdefault(key, {})[ijk] = alias_comp

    if not freq_map and not alias_map:
        return None

    # Resolve within-frequency aliases (e.g., beta(X;Y,X) = beta(Y,X,X)).
    for key, aliases in alias_map.items():
        comp_map = freq_map.setdefault(key, {})
        pending = dict(aliases)
        for _ in range(len(pending) + 1):
            changed = False
            for comp, alias in list(pending.items()):
                if comp in comp_map:
                    pending.pop(comp, None)
                    continue
                if alias in comp_map:
                    comp_map[comp] = comp_map[alias]
                    pending.pop(comp, None)
                    changed = True
            if not changed:
                break

    # Fill remaining components by swapping B/C frequencies and j/k indices when possible.
    all_components: set[str] = set()
    for comp_map in freq_map.values():
        all_components.update(comp_map.keys())
    for aliases in alias_map.values():
        all_components.update(aliases.keys())
        all_components.update(aliases.values())

    for key, comp_map in freq_map.items():
        a, b, c = key
        swap_key = (a, c, b)
        if swap_key not in freq_map:
            continue
        swap_map = freq_map[swap_key]
        for comp in all_components:
            if comp in comp_map:
                continue
            if len(comp) != 3:
                continue
            swap_comp = f"{comp[0]}{comp[2]}{comp[1]}"
            if swap_comp in swap_map:
                comp_map[comp] = swap_map[swap_comp]

    freq_list = sorted(freq_map.keys())
    components = sorted({comp for entry in freq_map.values() for comp in entry.keys()})

    values = np.full((len(freq_list), len(components)), np.nan, dtype=float)
    for i, freq in enumerate(freq_list):
        for j, comp in enumerate(components):
            if comp in freq_map[freq]:
                values[i, j] = float(freq_map[freq][comp])

    return {
        "omega": np.asarray(freq_list, dtype=float),
        "components": components,
        "values": values,
        "shape": ("freq", "component"),
    }


def _normalize_raman_rows(
    raman_by_freq: dict[float, list[dict[str, Any]]]
) -> dict[float, list[dict[str, Any]]]:
    for _, rows in raman_by_freq.items():
        for row in rows:
            if "dep_ratio" not in row and "depol_ratio" in row:
                row["dep_ratio"] = row.get("depol_ratio")
        rows.sort(key=lambda r: (float(r.get("freq_cm1", 0.0)), int(r.get("mode", 0))))
        for idx, row in enumerate(rows, start=1):
            row["mode"] = idx
    return raman_by_freq


def _parse_raman_from_legacy(lines: list[str]) -> dict[str, Any] | None:
    legacy = LegacyDaltonParser(lines)
    raman_by_freq = legacy.parse_raman_tables() or {}
    vib_rows = legacy.parse_vibrations() or []
    vib_freqs = [row.get("freq_cm1") for row in vib_rows if "freq_cm1" in row]

    try:
        legacy.parse_geometry()
    except Exception:
        pass

    try:
        legacy.parse_num_frequencies()
    except Exception:
        pass

    polar_cart = None
    polar_normal = None
    try:
        polar_cart = legacy.parse_polar_cartesian_gradients()
    except Exception:
        polar_cart = None

    try:
        polar_normal = legacy.parse_polar_normal_gradients()
    except Exception:
        polar_normal = None

    pol_freqs = sorted(raman_by_freq.keys()) if raman_by_freq else []
    if not pol_freqs:
        if polar_cart:
            pol_freqs = sorted(polar_cart.keys())
        elif polar_normal:
            pol_freqs = sorted(polar_normal.keys())

    polar_derivatives: list[np.ndarray] = []
    if polar_cart and pol_freqs:
        for f in pol_freqs:
            if f in polar_cart:
                polar_derivatives.append(polar_cart[f])

    polar_derivatives_by_mode: list[np.ndarray] = []
    if polar_normal and pol_freqs:
        for f in pol_freqs:
            if f in polar_normal:
                polar_derivatives_by_mode.append(polar_normal[f])

    if not (raman_by_freq and vib_freqs and pol_freqs):
        return None

    raman_by_freq = _normalize_raman_rows(raman_by_freq)
    return {
        "polarization_frequencies": np.asarray(pol_freqs, dtype=float),
        "vibrational_frequencies": np.asarray(vib_freqs, dtype=float),
        "polarizability_derivatives": polar_derivatives,
        "polarizability_derivatives_by_mode": polar_derivatives_by_mode,
        "raman_by_freq": raman_by_freq,
    }


def _sorted_out_files(root: Path, *, primary: Path | None = None) -> list[Path]:
    out_files = sorted(root.glob("*.out"))
    dalton_upper = root / "DALTON.OUT"
    if dalton_upper.exists() and dalton_upper not in out_files:
        out_files.insert(0, dalton_upper)

    if primary is not None:
        primary = Path(primary).expanduser().resolve()
        if primary.exists() and primary in out_files:
            out_files.remove(primary)
            out_files.insert(0, primary)

    return out_files


def _parse_one_out(path: Path) -> dict[str, Any]:
    lines = _read_lines(path)
    parser = DaltonParser(lines)

    out: dict[str, Any] = {
        "out_path": str(path),
        "label": None,
        "basis_guess": None,
        "basis": None,
        "ground_state_energy": None,
        "molecule": None,
        "molecule_source": None,
        "alpha": None,
        "beta": None,
        "raman": None,
        "excited_states": None,
        "warnings": [],
    }

    label, basis_guess = split_label_basis_from_outname(path.name)
    out["label"] = label
    out["basis_guess"] = basis_guess

    basis = _infer_basis_from_content(lines)
    if basis:
        out["basis"] = basis
    elif basis_guess:
        out["basis"] = basis_guess

    try:
        mol = parser.parse_geometry()
        out["molecule"] = mol
        out["molecule_source"] = "out"
        if out.get("basis") is None and parser.mol_block_basis:
            out["basis"] = parser.mol_block_basis
    except Exception as exc:
        out["warnings"].append(f"Failed to parse geometry ({type(exc).__name__}: {exc})")

    try:
        out["ground_state_energy"] = parser.parse_final_hf_energy()
    except Exception:
        pass

    if _DALTONPROJECT_AVAILABLE:
        try:
            dp_parser = _DaltonProjectOutputParser(str(path.with_suffix('')))
            pol = dp_parser.polarizabilities
            out["alpha"] = _alpha_tensor_to_data(pol.values, pol.frequencies.tolist())
        except Exception:
            try:
                tensors, freqs, _ = parse_polarizability_section(lines, start_index=0)
                out["alpha"] = _alpha_tensor_to_data(tensors, freqs)
            except Exception:
                pass
    else:
        try:
            tensors, freqs, _ = parse_polarizability_section(lines, start_index=0)
            out["alpha"] = _alpha_tensor_to_data(tensors, freqs)
        except Exception:
            pass

    try:
        beta = _parse_beta_from_quad_lines(lines)
        if beta is not None:
            out["beta"] = beta
    except Exception:
        pass

    try:
        raman = _parse_raman_from_legacy(lines)
        if raman is not None:
            out["raman"] = raman
    except Exception:
        pass

    try:
        excited_states = parse_electronic_excitations(lines)
        if excited_states:
            out["excited_states"] = excited_states
    except Exception:
        pass

    return out


def parse_run(calc: Calculation) -> None:
    calc.meta["style"] = "dalton"

    primary_out = (
        calc.artifacts.get("output")
        or calc.artifacts.get("out")
        or calc.artifacts.get("dalton_out")
    )

    out_files = _sorted_out_files(calc.root, primary=primary_out)
    if not out_files and primary_out is not None and primary_out.exists():
        out_files = [primary_out]
    if not out_files:
        return

    calc.meta["primary_out"] = str(out_files[0])
    calc.meta["out_files"] = [p.name for p in out_files]

    outputs: dict[str, Any] = {}
    alpha_by_out: dict[str, Any] = {}
    beta_by_out: dict[str, Any] = {}
    raman_by_out: dict[str, Any] = {}
    excited_by_out: dict[str, list[dict[str, Any]]] = {}
    energy_by_out: dict[str, float] = {}
    basis_by_out: dict[str, str] = {}
    warnings: list[str] = []

    for out_path in out_files:
        parsed = _parse_one_out(out_path)
        outputs[out_path.name] = parsed
        for w in parsed.get("warnings") or []:
            warnings.append(f"{out_path.name}: {w}")

        if parsed.get("alpha") is not None:
            alpha_by_out[out_path.name] = parsed["alpha"]
        if parsed.get("beta") is not None:
            beta_by_out[out_path.name] = parsed["beta"]
        if parsed.get("raman") is not None:
            raman_by_out[out_path.name] = parsed["raman"]
        excited_rows = parsed.get("excited_states")
        if isinstance(excited_rows, list) and excited_rows:
            excited_by_out[out_path.name] = excited_rows
        if parsed.get("ground_state_energy") is not None:
            energy_by_out[out_path.name] = float(parsed["ground_state_energy"])
        if parsed.get("basis") is not None:
            basis_by_out[out_path.name] = str(parsed["basis"])

        if calc.molecule is None and parsed.get("molecule") is not None:
            calc.molecule = parsed["molecule"]
            calc.meta.setdefault("molecule_source", parsed.get("molecule_source") or "out")
            calc.meta.setdefault("molecule_path", str(out_path))

    calc.data["dalton_outputs"] = outputs

    primary_name = out_files[0].name
    primary = outputs.get(primary_name) or {}

    if primary.get("label"):
        calc.meta["label"] = primary["label"]
        calc.meta["molecule"] = primary["label"]
        calc.meta["molecule_key"] = primary["label"]
        calc.meta.setdefault("inferred_from", {}).setdefault("molecule", "filename")

    if primary.get("basis_guess"):
        calc.meta["basis_guess"] = primary["basis_guess"]

    if primary.get("basis") is not None:
        calc.basis = primary["basis"]
        calc.meta["basis"] = calc.basis
        inferred = calc.meta.setdefault("inferred_from", {})
        if primary.get("basis") == primary.get("basis_guess"):
            inferred.setdefault("basis", "filename")
        else:
            inferred.setdefault("basis", "content")
    else:
        calc.meta.setdefault("warnings", []).append("Basis not found in output or filename.")

    if energy_by_out:
        if primary_name in energy_by_out:
            calc.meta["ground_state_energy"] = energy_by_out[primary_name]
        else:
            calc.meta["ground_state_energy_by_out"] = energy_by_out

    if alpha_by_out:
        if len(alpha_by_out) == 1:
            calc.data["alpha"] = next(iter(alpha_by_out.values()))
        elif primary_name in alpha_by_out:
            calc.data["alpha"] = alpha_by_out[primary_name]
            calc.data["alpha_by_out"] = alpha_by_out
        else:
            calc.data["alpha_by_out"] = alpha_by_out

    if beta_by_out:
        if len(beta_by_out) == 1:
            calc.data["beta"] = next(iter(beta_by_out.values()))
        elif primary_name in beta_by_out:
            calc.data["beta"] = beta_by_out[primary_name]
            calc.data["beta_by_out"] = beta_by_out
        else:
            calc.data["beta_by_out"] = beta_by_out

    if raman_by_out:
        if len(raman_by_out) == 1:
            calc.data["raman"] = next(iter(raman_by_out.values()))
        elif primary_name in raman_by_out:
            calc.data["raman"] = raman_by_out[primary_name]
            calc.data["raman_by_out"] = raman_by_out
        else:
            calc.data["raman_by_out"] = raman_by_out

    if excited_by_out:
        if len(excited_by_out) == 1:
            calc.data["excited_states"] = next(iter(excited_by_out.values()))
        elif primary_name in excited_by_out:
            calc.data["excited_states"] = excited_by_out[primary_name]
            calc.data["excited_states_by_out"] = excited_by_out
        else:
            calc.data["excited_states_by_out"] = excited_by_out

    if basis_by_out and len(set(basis_by_out.values())) > 1:
        calc.meta["basis_by_out"] = basis_by_out

    if warnings:
        calc.meta.setdefault("warnings", []).extend(warnings)
