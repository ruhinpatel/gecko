from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from gecko.core.model import Calculation
from gecko.plugins.dalton.legacy.dalton import DaltonParser, parse_polarizability_section


_BETA_LINE_PREFIX = "@ B-freq"


def _read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8", errors="ignore").splitlines()


def _infer_basis_from_content(lines: list[str]) -> str | None:
    import re

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
    import re

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
    import re

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

    for line in lines[start_idx:]:
        if not line.startswith(_BETA_LINE_PREFIX):
            continue
        match = pattern.search(line)
        if not match:
            continue
        b_freq, c_freq, a1, a2, a3, value_raw = match.groups()
        try:
            value = float(str(value_raw).split()[0])
        except ValueError:
            continue

        b = float(b_freq)
        c = float(c_freq)
        a = -(b + c)
        ijk = f"{a1}{a2}{a3}".lower()
        key = (a, b, c)
        freq_map.setdefault(key, {})[ijk] = value

    if not freq_map:
        return None

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


def parse_run(calc: Calculation) -> None:
    out_path = (
        calc.artifacts.get("output")
        or calc.artifacts.get("out")
        or calc.artifacts.get("dalton_out")
    )
    if out_path is None or not out_path.exists():
        return

    calc.meta["style"] = "dalton"
    calc.data["dalton_out_path"] = str(out_path)

    inferred_from: dict[str, str] = {}
    lines = _read_lines(out_path)

    label, basis_guess = split_label_basis_from_outname(out_path.name)
    if label:
        calc.meta["label"] = label
        calc.meta["molecule_key"] = label
        calc.meta["molecule"] = label
        inferred_from.setdefault("molecule", "filename")
    if basis_guess:
        calc.meta["basis_guess"] = basis_guess
    basis = _infer_basis_from_content(lines)
    if basis:
        calc.meta["basis"] = basis
        inferred_from["basis"] = "content"

    if "basis" not in calc.meta and basis_guess:
        calc.meta["basis"] = basis_guess
        inferred_from.setdefault("basis", "filename")
    if "basis" not in calc.meta:
        calc.meta["basis"] = None
        calc.meta.setdefault("warnings", []).append("Basis not found in output or filename.")

    if inferred_from:
        calc.meta["inferred_from"] = inferred_from

    parser = DaltonParser(out_path)

    calc.molecule = parser.parse_geometry()
    if calc.molecule is not None:
        calc.meta.setdefault("molecule_source", "out")
        calc.meta.setdefault("molecule_path", str(out_path))

    if "basis" not in calc.meta or calc.meta.get("basis") is None:
        if parser.mol_block_basis:
            calc.meta["basis"] = parser.mol_block_basis
            calc.meta.setdefault("inferred_from", {}).setdefault("basis", "mol_block")

    try:
        calc.meta["ground_state_energy"] = parser.parse_final_hf_energy()
    except Exception:
        pass

    try:
        tensors, freqs, _ = parse_polarizability_section(parser.lines, start_index=0)
        calc.data["alpha"] = _alpha_tensor_to_data(tensors, freqs)
    except Exception:
        pass

    quad_path = calc.artifacts.get("dalton_quad_out") or out_path
    if quad_path and quad_path.exists():
        try:
            beta = _parse_beta_from_quad_lines(_read_lines(quad_path))
            if beta is not None:
                calc.data["beta"] = beta
        except Exception:
            pass
