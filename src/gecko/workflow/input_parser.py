"""Parser for MADNESS .in files → MadnessInputFile models.

Mirrors MADNESS's own parsing logic from ``QCCalculationParametersBase::read_internal``:

1. Section splitting — scan for ``dft``/``response``/``molecule`` tags and ``end``
2. Line processing — strip ``#`` comments, replace ``=`` with space, lowercase keys
3. Molecule section — lines where first token is a known element symbol → Atom
4. Type coercion — driven by the Pydantic model's field type annotations
5. Strict validation — Pydantic's ``extra="forbid"`` rejects unknown keys
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, get_args, get_origin, Literal

from gecko.workflow.input_model import (
    Atom,
    DFTSection,
    MadnessInputFile,
    MoleculeSection,
    ResponseSection,
)

# Element symbols for detecting atom coordinate lines in the molecule section
_ELEMENT_SYMBOLS = {
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
    "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
    "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Kr",
    "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
    "In", "Sn", "Sb", "Te", "I", "Xe",
    "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
    "Tl", "Pb", "Bi", "Po", "At", "Rn",
    "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
    "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn",
    "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
}


def parse_madness_input(text: str) -> MadnessInputFile:
    """Parse MADNESS .in file text into a ``MadnessInputFile`` model."""
    sections = _split_sections(text)
    atoms: list[Atom] = []

    dft_data = _parse_section_lines(sections.get("dft", []), DFTSection)
    response_data = _parse_section_lines(sections.get("response", []), ResponseSection)
    molecule_data, atoms = _parse_molecule_section(sections.get("molecule", []))

    dft = DFTSection.model_validate(dft_data)
    response = ResponseSection.model_validate(response_data)
    molecule = MoleculeSection.model_validate(molecule_data)

    return MadnessInputFile(dft=dft, molecule=molecule, response=response, atoms=atoms)


def parse_madness_input_file(path: Path) -> MadnessInputFile:
    """Parse a MADNESS .in file from disk."""
    return parse_madness_input(path.read_text())


# ---------------------------------------------------------------------------
# Section splitting
# ---------------------------------------------------------------------------

_SECTION_NAMES = {"dft", "response", "molecule"}


def _split_sections(text: str) -> dict[str, list[str]]:
    """Split input text into named sections.

    Returns a dict mapping section name → list of content lines (without the
    section header and ``end`` terminator).
    """
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    current_lines: list[str] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        # Strip comments
        if "#" in line:
            line = line[:line.index("#")].strip()
        if not line:
            continue

        token = line.split()[0].lower()

        if token in _SECTION_NAMES and current_section is None:
            current_section = token
            current_lines = []
        elif token == "end" and current_section is not None:
            sections[current_section] = current_lines
            current_section = None
            current_lines = []
        elif current_section is not None:
            current_lines.append(line)

    return sections


# ---------------------------------------------------------------------------
# Line parsing
# ---------------------------------------------------------------------------


def _parse_section_lines(
    lines: list[str],
    model_cls: type,
) -> dict[str, Any]:
    """Parse key-value lines into a dict suitable for Pydantic model_validate.

    Uses the model's field annotations to drive type coercion.
    """
    data: dict[str, Any] = {}
    # Build a lookup: MADNESS key (alias or field name) → (field_name, annotation)
    key_map = _build_key_map(model_cls)

    for line in lines:
        # Replace = with space for key=value syntax
        line = line.replace("=", " ")
        parts = line.split(None, 1)
        if not parts:
            continue

        madness_key = parts[0].lower()
        raw_value = parts[1].strip() if len(parts) > 1 else "true"

        # Strip surrounding quotes from string values
        if raw_value.startswith('"') and raw_value.endswith('"'):
            raw_value = raw_value[1:-1]

        if madness_key not in key_map:
            raise ValueError(
                f"Unknown parameter {madness_key!r} in {model_cls.__name__}. "
                f"Valid keys: {sorted(key_map.keys())}"
            )

        field_name, annotation = key_map[madness_key]
        data[field_name] = _coerce_from_string(raw_value, annotation)

    return data


def _parse_molecule_section(
    lines: list[str],
) -> tuple[dict[str, Any], list[Atom]]:
    """Parse molecule section: parameters + atom coordinate lines."""
    atoms: list[Atom] = []
    param_lines: list[str] = []

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        # Check if first token is an element symbol
        first = parts[0]
        # Element symbols: first letter uppercase, rest lowercase
        if first.capitalize() in _ELEMENT_SYMBOLS and len(parts) >= 4:
            try:
                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                atoms.append(Atom(symbol=first.capitalize(), x=x, y=y, z=z))
                continue
            except (ValueError, IndexError):
                pass
        param_lines.append(line)

    data = _parse_section_lines(param_lines, MoleculeSection)
    return data, atoms


# ---------------------------------------------------------------------------
# Key map + type coercion
# ---------------------------------------------------------------------------


def _build_key_map(model_cls: type) -> dict[str, tuple[str, Any]]:
    """Build a mapping from MADNESS keys to (field_name, annotation) pairs."""
    key_map: dict[str, tuple[str, Any]] = {}
    for field_name, field_info in model_cls.model_fields.items():
        annotation = field_info.annotation
        alias = field_info.alias or field_name
        # Map both the alias (MADNESS key) and the Python field name
        key_map[alias.lower()] = (field_name, annotation)
        key_map[field_name.lower()] = (field_name, annotation)
    return key_map


def _coerce_from_string(raw: str, annotation: Any) -> Any:
    """Coerce a raw string value to the target type annotation."""
    origin = get_origin(annotation)
    args = get_args(annotation)

    # Literal types
    if origin is Literal:
        val = raw.lower()
        allowed = [a.lower() if isinstance(a, str) else a for a in args]
        if val not in allowed:
            raise ValueError(f"{raw!r} not in allowed values {args}")
        # Return the original-case version from the Literal
        for a in args:
            if isinstance(a, str) and a.lower() == val:
                return a
            if a == val:
                return a
        return val

    # list types
    if origin is list:
        elem_type = args[0] if args else str
        return _parse_list_value(raw, elem_type)

    # Scalar types
    if annotation is bool or annotation == bool:
        return raw.lower() in ("true", "1", "yes")

    if annotation is int or annotation == int:
        return int(float(raw))  # handle "1e-06" → 0 edge case

    if annotation is float or annotation == float:
        return float(raw)

    # str
    return raw


def _parse_list_value(raw: str, elem_type: type) -> list:
    """Parse a MADNESS list value like ``[0.0,0.05]`` or ``[polarizability,raman]``."""
    stripped = raw.strip()
    if stripped.startswith("[") and stripped.endswith("]"):
        inner = stripped[1:-1]
    else:
        inner = stripped

    if not inner.strip():
        return []

    items = [x.strip().strip('"') for x in inner.split(",")]

    if elem_type is float:
        return [float(x) for x in items]
    if elem_type is int:
        return [int(float(x)) for x in items]
    if elem_type is bool:
        return [x.lower() in ("true", "1", "yes") for x in items]
    return items
