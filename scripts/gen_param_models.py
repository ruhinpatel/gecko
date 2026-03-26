#!/usr/bin/env python3
"""Codegen: parse MADNESS C++ headers and generate Pydantic v2 models.

Parses ``initialize<Type>("key", default, "description", {allowed_values})``
calls from three C++ headers and emits ``src/gecko/workflow/input_model.py``
with typed, validated Pydantic models that mirror MADNESS parameters 1:1.

Usage::

    python scripts/gen_param_models.py
    python scripts/gen_param_models.py --madness-src /path/to/madness
    python scripts/gen_param_models.py --output src/gecko/workflow/input_model.py
"""

from __future__ import annotations

import argparse
import re
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_MADNESS_SRC = Path(__file__).resolve().parent.parent.parent / (
    "madness-worktrees/molresponse-feature-next"
)

_HEADER_PATHS = {
    "dft": "src/madness/chem/CalculationParameters.h",
    "response": "src/madness/chem/ResponseParameters.hpp",
    "molecule": "src/madness/chem/molecule.h",
}

# Parameters that ResponseParameters inherits from QCCalculationParametersBase
# but uses via getter methods. These are not defined via initialize<>() in
# ResponseParameters.hpp but appear in real input files.
_RESPONSE_INHERITED_PARAMS: list  # populated after Param class is defined

# ---------------------------------------------------------------------------
# C++ type → Python type mapping
# ---------------------------------------------------------------------------

_CPP_TO_PYTHON: dict[str, str] = {
    "std::string": "str",
    "double": "float",
    "int": "int",
    "size_t": "int",
    "bool": "bool",
    "std::vector<double>": "list[float]",
    "std::vector<std::string>": "list[str]",
    "std::vector<int>": "list[int]",
}


# ---------------------------------------------------------------------------
# Parsed parameter
# ---------------------------------------------------------------------------


@dataclass
class Param:
    cpp_type: str
    key: str
    default: str  # raw C++ literal
    description: str
    allowed: list[str] = field(default_factory=list)

    @property
    def python_type(self) -> str:
        return _CPP_TO_PYTHON.get(self.cpp_type, "str")

    @property
    def python_field_name(self) -> str:
        """Convert dotted MADNESS key to a valid Python identifier."""
        return self.key.replace(".", "_")

    @property
    def needs_alias(self) -> bool:
        return "." in self.key

    def python_default(self) -> str:
        """Convert C++ default literal to Python repr."""
        py_type = self.python_type

        if py_type == "bool":
            return "True" if self.default == "true" else "False"

        if py_type == "str":
            # Strip quotes if present, add Python quotes
            val = self.default.strip('"')
            return repr(val)

        if py_type == "float":
            # Normalise C++ float literals
            val = self.default.replace("e-0", "e-").replace("e+0", "e+")
            try:
                return repr(float(val))
            except ValueError:
                return repr(0.0)

        if py_type == "int":
            try:
                return repr(int(self.default))
            except ValueError:
                return repr(0)

        if py_type.startswith("list["):
            return self._parse_list_default()

        return repr(self.default)

    def _parse_list_default(self) -> str:
        raw = self.default.strip()
        # Handle C++ empty vector constructor: std::vector<T>()
        if raw.endswith("()") or raw == "{}":
            return "[]"
        inner = raw.strip("{}")
        if not inner:
            return "[]"
        items = [x.strip().strip('"') for x in inner.split(",")]
        elem_type = self.python_type[5:-1]  # e.g. "float" from "list[float]"
        if elem_type == "float":
            return "[" + ", ".join(str(float(x)) for x in items) + "]"
        if elem_type == "int":
            return "[" + ", ".join(str(int(x)) for x in items) + "]"
        # str
        return "[" + ", ".join(repr(x) for x in items) + "]"

    @property
    def is_inherited(self) -> bool:
        return False

    def literal_type(self) -> str | None:
        """Return a Literal[...] type string if allowed values exist."""
        if not self.allowed:
            return None
        vals = ", ".join(repr(v) for v in self.allowed)
        return f"Literal[{vals}]"


# ---------------------------------------------------------------------------
# Parser: extract initialize<>() calls from C++ source
# ---------------------------------------------------------------------------


# Now populate _RESPONSE_INHERITED_PARAMS after Param is defined
_RESPONSE_INHERITED_PARAMS = [
    Param(cpp_type="int", key="maxiter", default="25", description="maximum number of iterations"),
    Param(cpp_type="std::vector<double>", key="protocol", default="{1.e-4,1.e-6}", description="calculation protocol"),
    Param(cpp_type="std::string", key="deriv", default="abgv", description="derivative method",
          allowed=["abgv", "bspline", "ble"]),
    Param(cpp_type="std::string", key="dft_deriv", default="abgv", description="derivative method for gga potentials",
          allowed=["abgv", "bspline", "ble"]),
    Param(cpp_type="int", key="k", default="-1", description="polynomial order"),
    Param(cpp_type="bool", key="save", default="false", description="save response orbitals to disk"),
    Param(cpp_type="bool", key="restart", default="false", description="restart from saved response orbitals"),
]


def _collapse_cpp_string_concat(text: str) -> str:
    """Collapse C++ adjacent string literals: ``"foo" "bar"`` → ``"foobar"``."""
    return re.sub(r'"\s+"', "", text)


def _extract_params(source: str) -> list[Param]:
    """Extract all initialize<>() calls from a C++ source string."""
    # Join all lines and collapse C++ string concatenation
    source = source.replace("\n", " ")
    source = _collapse_cpp_string_concat(source)

    # Find all initialize<...>(...) calls by matching balanced angle brackets
    pattern = re.compile(r'initialize\s*<')
    params: list[Param] = []

    for m in pattern.finditer(source):
        # Match balanced <> to extract the type
        start = m.end()
        depth = 1
        j = start
        while j < len(source) and depth > 0:
            if source[j] == "<":
                depth += 1
            elif source[j] == ">":
                depth -= 1
            j += 1
        if depth != 0:
            continue
        cpp_type = re.sub(r"\s+", "", source[start:j - 1].strip())
        # Skip to the opening paren
        rest = source[j:].lstrip()
        if not rest.startswith("("):
            continue
        # Adjust m.end() equivalent to after the (
        paren_start = j + source[j:].index("(") + 1
        # Find the matching closing paren
        depth = 1
        i = paren_start
        while i < len(source) and depth > 0:
            if source[i] == "(":
                depth += 1
            elif source[i] == ")":
                depth -= 1
            i += 1
        if depth != 0:
            continue
        args_str = source[paren_start:i - 1].strip()

        # Parse the arguments: "key", default, "description" [, {allowed}]
        parsed = _parse_init_args(args_str)
        if parsed is None:
            continue

        key, default_raw, desc, allowed = parsed
        params.append(Param(
            cpp_type=cpp_type,
            key=key,
            default=default_raw,
            description=desc,
            allowed=allowed,
        ))
    return params


def _parse_init_args(args: str) -> tuple[str, str, str, list[str]] | None:
    """Parse the arguments of an initialize<>() call.

    Returns (key, default, description, allowed_values) or None on failure.
    """
    # Extract the key (first quoted string)
    key_match = re.match(r'\s*"([^"]+)"\s*,\s*', args)
    if not key_match:
        return None
    key = key_match.group(1)
    rest = args[key_match.end():]

    # Extract default value — everything up to the next comma followed by a quote
    # Handle braces for vector defaults like {0.0, 0.0, 0.0}
    default_raw, rest = _split_default_and_rest(rest)
    if rest is None:
        return None

    # Extract description (quoted string, possibly already collapsed)
    desc_match = re.match(r'\s*"((?:[^"\\]|\\.)*)"\s*', rest)
    if not desc_match:
        return None
    desc = desc_match.group(1).replace('\\"', '"')
    rest = rest[desc_match.end():]

    # Extract optional allowed values {"val1", "val2", ...}
    allowed: list[str] = []
    allowed_match = re.match(r'\s*,\s*\{([^}]*)\}', rest)
    if allowed_match:
        allowed = re.findall(r'"([^"]*)"', allowed_match.group(1))

    return key, default_raw, desc, allowed


def _split_default_and_rest(text: str) -> tuple[str, str | None]:
    """Split default value from the rest of the args.

    Handles brace-enclosed defaults like ``{0.0, 0.0, 0.0}``.
    """
    text = text.strip()
    if text.startswith("{"):
        # Find matching }
        depth = 0
        for i, ch in enumerate(text):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    default = text[:i + 1]
                    rest = text[i + 1:].lstrip()
                    if rest.startswith(","):
                        rest = rest[1:].lstrip()
                    return default, rest
        return text, None
    else:
        # Find comma before the description string
        # The default ends at the comma that is followed (possibly with spaces) by a quote
        # But we must skip commas inside the default itself
        match = re.match(r'(.+?)\s*,\s*"', text)
        if match:
            default = match.group(1).strip()
            rest = text[match.start() + len(match.group(0)) - 1:]  # back up to the quote
            return default, rest
        return text, None


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

_FILE_HEADER = '''\
"""Pydantic v2 models for MADNESS input file sections.

.. warning:: This file is **auto-generated** by ``scripts/gen_param_models.py``.
   Do not edit the generated sections by hand — re-run the codegen script instead.
   Hand-written additions between ``# --- BEGIN HAND-WRITTEN ---`` and
   ``# --- END HAND-WRITTEN ---`` markers are preserved across regeneration.

Generated: {timestamp}
Source headers:
  DFT:      {dft_header}
  Response: {response_header}
  Molecule: {molecule_header}
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


'''

_SECTION_CLASS_NAMES = {
    "dft": "DFTSection",
    "response": "ResponseSection",
    "molecule": "MoleculeSection",
}


def _generate_section_class(section: str, params: list[Param]) -> str:
    """Generate a Pydantic BaseModel class for a section."""
    class_name = _SECTION_CLASS_NAMES[section]
    lines = [
        f"class {class_name}(BaseModel):",
        f'    """Parameters for the ``{section}`` section of a MADNESS .in file."""',
        "",
        '    model_config = ConfigDict(populate_by_name=True, extra="forbid")',
        "",
    ]

    for p in params:
        py_type = p.python_type
        lit_type = p.literal_type()
        field_name = p.python_field_name
        default = p.python_default()

        # Build type annotation
        if lit_type:
            ann_type = lit_type
        else:
            ann_type = py_type

        # Build Field() kwargs
        field_kwargs: list[str] = []
        field_kwargs.append(f"default={default}")
        if p.needs_alias:
            field_kwargs.append(f'alias="{p.key}"')
        if p.description:
            # Truncate very long descriptions
            desc = p.description[:120]
            field_kwargs.append(f'description="{desc}"')

        field_str = f"Field({', '.join(field_kwargs)})"
        lines.append(f"    {field_name}: {ann_type} = {field_str}")
        lines.append("")

    return "\n".join(lines)


def _generate_full_module(
    dft_params: list[Param],
    response_params: list[Param],
    molecule_params: list[Param],
    header_paths: dict[str, str],
) -> str:
    """Generate the complete input_model.py source."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    header = _FILE_HEADER.format(
        timestamp=timestamp,
        dft_header=header_paths.get("dft", ""),
        response_header=header_paths.get("response", ""),
        molecule_header=header_paths.get("molecule", ""),
    )

    # Add known inherited parameters that ResponseParameters uses via getters
    # but doesn't initialize (inherited from QCCalculationParametersBase).
    _inherited_response_keys = {p.key for p in response_params}
    for inherited in _RESPONSE_INHERITED_PARAMS:
        if inherited.key not in _inherited_response_keys:
            response_params.append(inherited)

    dft_class = _generate_section_class("dft", dft_params)
    response_class = _generate_section_class("response", response_params)
    molecule_class = _generate_section_class("molecule", molecule_params)

    # Hand-written section for Atom, MadnessInputFile, etc.
    hand_written = textwrap.dedent('''\
        # --- BEGIN HAND-WRITTEN ---
        # Everything between BEGIN/END markers is preserved across codegen runs.


        class Atom(BaseModel):
            """Single atom entry in the molecule section."""

            symbol: str
            x: float
            y: float
            z: float


        # Extend MoleculeSection with atoms list (not from C++ headers)
        MoleculeSection.model_rebuild()


        class MadnessInputFile(BaseModel):
            """Complete MADNESS .in file represented as a Pydantic model.

            Sections default to empty (all MADNESS defaults).  Use ``from_file``
            to parse an existing file, or construct directly and call ``to_file``.
            """

            dft: DFTSection = Field(default_factory=DFTSection)
            molecule: MoleculeSection = Field(default_factory=MoleculeSection)
            response: ResponseSection = Field(default_factory=ResponseSection)
            atoms: list[Atom] = Field(default_factory=list, description="Atom coordinates for the molecule section")

            @classmethod
            def from_file(cls, path: str | Path) -> "MadnessInputFile":
                """Parse a MADNESS .in file into this model."""
                from gecko.workflow.input_parser import parse_madness_input_file

                return parse_madness_input_file(Path(path))

            def to_file(self, path: str | Path) -> None:
                """Serialize this model to a MADNESS .in file."""
                from gecko.workflow.input_serializer import serialize_madness_input

                Path(path).write_text(serialize_madness_input(self))

            def to_madness_str(self) -> str:
                """Return the MADNESS .in file content as a string."""
                from gecko.workflow.input_serializer import serialize_madness_input

                return serialize_madness_input(self)

            def get(self, dotted_key: str) -> Any:
                """Get a parameter value by dotted key, e.g. ``dft.xc`` or ``response.dipole.frequencies``.

                The first segment is the section name (dft/molecule/response).
                The rest is the parameter key within that section.
                """
                section_name, _, param_key = dotted_key.partition(".")
                if not param_key:
                    raise ValueError(f"Key must be section.param, got {dotted_key!r}")

                section = self._get_section(section_name)

                # Try field name first, then alias
                if hasattr(section, param_key.replace(".", "_")):
                    return getattr(section, param_key.replace(".", "_"))

                # Search by alias
                for fname, finfo in type(section).model_fields.items():
                    alias = finfo.alias or fname
                    if alias == param_key:
                        return getattr(section, fname)

                raise KeyError(f"Unknown parameter {param_key!r} in section {section_name!r}")

            def set(self, dotted_key: str, value: Any) -> None:
                """Set a parameter value by dotted key.

                The value is coerced to the field's type using the Pydantic model.
                """
                section_name, _, param_key = dotted_key.partition(".")
                if not param_key:
                    raise ValueError(f"Key must be section.param, got {dotted_key!r}")

                section = self._get_section(section_name)
                field_name = self._resolve_field_name(section, param_key)

                finfo = type(section).model_fields[field_name]
                coerced = _coerce_value(value, finfo.annotation)
                setattr(section, field_name, coerced)

            def _get_section(self, name: str) -> BaseModel:
                if name == "dft":
                    return self.dft
                if name == "response":
                    return self.response
                if name == "molecule":
                    return self.molecule
                raise KeyError(f"Unknown section {name!r}. Valid: dft, molecule, response")

            @staticmethod
            def _resolve_field_name(section: BaseModel, param_key: str) -> str:
                """Resolve a MADNESS parameter key to its Pydantic field name."""
                py_name = param_key.replace(".", "_")
                if py_name in type(section).model_fields:
                    return py_name
                # Search by alias
                for fname, finfo in type(section).model_fields.items():
                    alias = finfo.alias or fname
                    if alias == param_key:
                        return fname
                raise KeyError(f"Unknown parameter {param_key!r}")


        def _coerce_value(value: Any, annotation: Any) -> Any:
            """Coerce a string value to the target type annotation."""
            if not isinstance(value, str):
                return value

            import typing

            origin = getattr(annotation, "__origin__", None)
            args = getattr(annotation, "__args__", ())

            # Handle typing.Literal
            if origin is Literal:
                if value in args:
                    return value
                raise ValueError(f"{value!r} not in allowed values {args}")

            # Handle list types
            if origin is list:
                elem_type = args[0] if args else str
                # Parse [x,y,z] syntax
                stripped = value.strip()
                if stripped.startswith("[") and stripped.endswith("]"):
                    inner = stripped[1:-1]
                else:
                    inner = stripped
                if not inner.strip():
                    return []
                items = [x.strip() for x in inner.split(",")]
                return [_coerce_scalar(x, elem_type) for x in items]

            # Handle Optional / Union types (Literal is already caught above)
            if origin is typing.Union:
                for arg in args:
                    if arg is type(None):
                        continue
                    try:
                        return _coerce_value(value, arg)
                    except (ValueError, TypeError):
                        continue
                return value

            # Scalar types
            if annotation is bool or annotation == bool:
                return value.lower() in ("true", "1", "yes")
            if annotation is int or annotation == int:
                return int(value)
            if annotation is float or annotation == float:
                return float(value)
            return value


        def _coerce_scalar(value: str, target_type: type) -> Any:
            if target_type is float:
                return float(value)
            if target_type is int:
                return int(value)
            if target_type is bool:
                return value.lower() in ("true", "1", "yes")
            return value


        # --- END HAND-WRITTEN ---
    ''')

    return header + dft_class + "\n\n" + response_class + "\n\n" + molecule_class + "\n\n" + hand_written


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Pydantic models from MADNESS C++ headers")
    parser.add_argument(
        "--madness-src",
        type=Path,
        default=_DEFAULT_MADNESS_SRC,
        help="Path to MADNESS source root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "src/gecko/workflow/input_model.py",
        help="Output path for generated module",
    )
    args = parser.parse_args()

    header_paths: dict[str, str] = {}
    all_params: dict[str, list[Param]] = {}

    for section, rel_path in _HEADER_PATHS.items():
        full_path = args.madness_src / rel_path
        if not full_path.exists():
            raise FileNotFoundError(f"Header not found: {full_path}")
        header_paths[section] = str(full_path)
        source = full_path.read_text()
        params = _extract_params(source)
        all_params[section] = params
        print(f"  {section}: parsed {len(params)} parameters from {rel_path}")

    output = _generate_full_module(
        dft_params=all_params["dft"],
        response_params=all_params["response"],
        molecule_params=all_params["molecule"],
        header_paths=header_paths,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(output)
    print(f"\nGenerated {args.output} ({len(output)} bytes)")


if __name__ == "__main__":
    main()
