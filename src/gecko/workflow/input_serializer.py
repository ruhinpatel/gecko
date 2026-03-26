"""Serializer: MadnessInputFile → MADNESS .in file text.

Converts a ``MadnessInputFile`` model back to the MADNESS input format.
Reuses ``_render_value()`` from ``params.py`` for consistent value formatting.
"""

from __future__ import annotations

from gecko.workflow.input_model import Atom, MadnessInputFile
from gecko.workflow.params import _render_value


def serialize_madness_input(inp: MadnessInputFile) -> str:
    """Serialize a ``MadnessInputFile`` to MADNESS .in format."""
    parts: list[str] = []

    # DFT section
    dft_lines = _serialize_section(inp.dft, type(inp.dft).model_fields)
    if dft_lines:
        parts.append("dft")
        parts.extend(f"    {line}" for line in dft_lines)
        parts.append("end")

    # Response section
    resp_lines = _serialize_section(inp.response, type(inp.response).model_fields)
    if resp_lines:
        parts.append("response")
        parts.extend(f"    {line}" for line in resp_lines)
        parts.append("end")

    # Molecule section
    mol_lines = _serialize_section(inp.molecule, type(inp.molecule).model_fields)
    atom_lines = _serialize_atoms(inp.atoms)
    if mol_lines or atom_lines:
        parts.append("molecule")
        parts.extend(f"    {line}" for line in mol_lines)
        parts.extend(f"    {line}" for line in atom_lines)
        parts.append("end")

    return "\n".join(parts) + "\n"


def _serialize_section(
    section: object,
    fields: dict,
) -> list[str]:
    """Serialize a section model to key-value lines.

    Skips fields that equal their default, to keep the output minimal.
    """
    lines: list[str] = []
    defaults = section.__class__()

    for field_name, field_info in fields.items():
        value = getattr(section, field_name)
        default_value = getattr(defaults, field_name)

        # Skip values that match the default
        if value == default_value:
            continue

        # Use the alias (MADNESS key) if available, otherwise field name
        madness_key = field_info.alias or field_name
        rendered = _render_value(value)

        # Quote string values that contain no special chars (matching MADNESS convention)
        if isinstance(value, str):
            rendered = f'"{value}"'

        lines.append(f"{madness_key} {rendered}")

    return lines


def _serialize_atoms(atoms: list[Atom]) -> list[str]:
    """Serialize atom coordinate lines."""
    lines: list[str] = []
    for atom in atoms:
        lines.append(f"{atom.symbol} {atom.x:.12f} {atom.y:.12f} {atom.z:.12f}")
    return lines
