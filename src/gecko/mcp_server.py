"""Gecko MCP Server — exposes gecko's input file, geometry, and analysis tools.

Run standalone::

    python -m gecko.mcp_server

Or configure in .mcp.json for Claude Code integration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("gecko")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_MOL_LIB = Path(
    os.environ.get(
        "GECKO_MOL_LIB",
        "/gpfs/projects/rjh/adrian/development/madness-worktrees/molecules",
    )
)

_SCRATCH = Path(
    os.environ.get(
        "GECKO_SCRATCH",
        "/gpfs/scratch/ahurtado/gecko_calcs",
    )
)

# ---------------------------------------------------------------------------
# Tools — Input File Operations
# ---------------------------------------------------------------------------


@server.tool()
def parse_input(path: str) -> str:
    """Parse a MADNESS .in file and return all parameters as JSON.

    Args:
        path: Absolute or relative path to the .in file.
    """
    from gecko.workflow.input_model import MadnessInputFile

    inp = MadnessInputFile.from_file(path)
    data = inp.model_dump(by_alias=True)
    return json.dumps(data, indent=2)


@server.tool()
def show_input(path: str, section: str = "", format: str = "madness") -> str:
    """Display a MADNESS .in file in human-readable or JSON format.

    Args:
        path: Path to the .in file.
        section: Optional section filter: "dft", "response", or "molecule". Empty = all.
        format: Output format: "madness" (default) or "json".
    """
    from gecko.workflow.input_model import MadnessInputFile
    from gecko.workflow.input_serializer import _serialize_section, _serialize_atoms

    inp = MadnessInputFile.from_file(path)

    if format == "json":
        if section:
            sec = inp._get_section(section)
            data = sec.model_dump(by_alias=True)
            if section == "molecule":
                data["atoms"] = [a.model_dump() for a in inp.atoms]
        else:
            data = inp.model_dump(by_alias=True)
        return json.dumps(data, indent=2)

    if section:
        sec = inp._get_section(section)
        lines = _serialize_section(sec, type(sec).model_fields)
        parts = [section]
        parts.extend(f"    {line}" for line in lines)
        if section == "molecule":
            for line in _serialize_atoms(inp.atoms):
                parts.append(f"    {line}")
        parts.append("end")
        return "\n".join(parts)

    return inp.to_madness_str()


@server.tool()
def get_parameter(path: str, key: str) -> str:
    """Get a single parameter value from a MADNESS .in file.

    Args:
        path: Path to the .in file.
        key: Dotted key like "dft.xc", "response.dipole.frequencies", "molecule.units".
    """
    from gecko.workflow.input_model import MadnessInputFile
    from gecko.workflow.params import _render_value

    inp = MadnessInputFile.from_file(path)
    return _render_value(inp.get(key))


@server.tool()
def set_parameter(path: str, key: str, value: str, dry_run: bool = False) -> str:
    """Set a parameter in a MADNESS .in file.

    Args:
        path: Path to the .in file.
        key: Dotted key like "dft.xc", "response.dipole.frequencies".
        value: New value as a string (auto-coerced to the correct type).
        dry_run: If True, return the modified content without writing to disk.
    """
    from gecko.workflow.input_model import MadnessInputFile

    inp = MadnessInputFile.from_file(path)
    inp.set(key, value)

    if dry_run:
        return inp.to_madness_str()

    inp.to_file(path)
    return f"Set {key}={value} in {path}"


@server.tool()
def validate_input(path: str) -> str:
    """Validate a MADNESS .in file against the parameter schema.

    Args:
        path: Path to the .in file.
    """
    from gecko.workflow.input_model import MadnessInputFile

    try:
        inp = MadnessInputFile.from_file(path)
        n_atoms = len(inp.atoms)
        dft_defaults = type(inp.dft)()
        resp_defaults = type(inp.response)()
        mol_defaults = type(inp.molecule)()
        n_dft = sum(1 for f in type(inp.dft).model_fields if getattr(inp.dft, f) != getattr(dft_defaults, f))
        n_resp = sum(1 for f in type(inp.response).model_fields if getattr(inp.response, f) != getattr(resp_defaults, f))
        n_mol = sum(1 for f in type(inp.molecule).model_fields if getattr(inp.molecule, f) != getattr(mol_defaults, f))
        return (
            f"Valid: {path}\n"
            f"  dft: {n_dft} non-default params\n"
            f"  response: {n_resp} non-default params\n"
            f"  molecule: {n_mol} non-default params, {n_atoms} atoms"
        )
    except Exception as exc:
        return f"Invalid: {path}\n  Error: {exc}"


@server.tool()
def diff_inputs(file1: str, file2: str) -> str:
    """Semantic parameter-level diff between two MADNESS .in files.

    Args:
        file1: Path to the first .in file.
        file2: Path to the second .in file.
    """
    from gecko.workflow.input_model import MadnessInputFile
    from gecko.workflow.params import _render_value

    inp1 = MadnessInputFile.from_file(file1)
    inp2 = MadnessInputFile.from_file(file2)
    diffs: list[str] = []

    for section_name in ("dft", "response", "molecule"):
        s1 = inp1._get_section(section_name)
        s2 = inp2._get_section(section_name)
        for field_name, field_info in type(s1).model_fields.items():
            v1, v2 = getattr(s1, field_name), getattr(s2, field_name)
            if v1 != v2:
                key = field_info.alias or field_name
                diffs.append(f"  {section_name}.{key}: {_render_value(v1)} -> {_render_value(v2)}")

    if len(inp1.atoms) != len(inp2.atoms):
        diffs.append(f"  atoms: {len(inp1.atoms)} -> {len(inp2.atoms)}")
    else:
        for i, (a1, a2) in enumerate(zip(inp1.atoms, inp2.atoms)):
            if a1 != a2:
                diffs.append(f"  atom[{i}]: {a1.symbol} -> {a2.symbol}")

    if diffs:
        return f"Differences:\n" + "\n".join(diffs)
    return "Files are semantically identical."


@server.tool()
def create_input(
    output_path: str,
    set_params: list[str] | None = None,
    from_file: str = "",
    geom_file: str = "",
    molecule: str = "",
) -> str:
    """Create a new MADNESS .in file from scratch or by modifying an existing one.

    Args:
        output_path: Where to write the new .in file.
        set_params: List of KEY=VALUE pairs (e.g. ["dft.xc=b3lyp", "dft.k=8"]).
        from_file: Optional base .in file to start from.
        geom_file: Optional path to .xyz or .mol geometry file.
        molecule: Optional molecule name to look up in the library.
    """
    from gecko.workflow.input_model import MadnessInputFile, Atom

    if from_file:
        inp = MadnessInputFile.from_file(from_file)
    else:
        inp = MadnessInputFile()

    if geom_file:
        from gecko.workflow.geometry import load_geometry_from_file

        mol = load_geometry_from_file(Path(geom_file))
        inp.molecule.units = "angstrom"
        import numpy as np
        geom_angstrom = np.array(mol.geometry).reshape(-1, 3) * 0.529177249
        inp.atoms = [
            Atom(symbol=sym, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            for sym, xyz in zip(mol.symbols, geom_angstrom)
        ]
    elif molecule:
        found = _find_molecule(molecule)
        if found:
            from gecko.workflow.geometry import load_geometry_from_file

            mol = load_geometry_from_file(found)
            inp.molecule.units = "angstrom"
            import numpy as np
            geom_angstrom = np.array(mol.geometry).reshape(-1, 3) * 0.529177249
            inp.atoms = [
                Atom(symbol=sym, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                for sym, xyz in zip(mol.symbols, geom_angstrom)
            ]

    for kv in (set_params or []):
        key, _, value = kv.partition("=")
        if value:
            inp.set(key.strip(), value.strip())

    inp.to_file(output_path)
    return f"Created {output_path} ({len(inp.atoms)} atoms)"


@server.tool()
def input_json_schema() -> str:
    """Return the full JSON Schema for MADNESS input files.

    Useful for GUI form generation and programmatic validation.
    """
    from gecko.workflow.input_model import MadnessInputFile

    return json.dumps(MadnessInputFile.model_json_schema(), indent=2)


# ---------------------------------------------------------------------------
# Tools — Geometry
# ---------------------------------------------------------------------------


@server.tool()
def list_molecules(pattern: str = "") -> str:
    """List molecules available in the local geometry library.

    Args:
        pattern: Optional substring filter (case-insensitive). Empty = list all.
    """
    mols: list[str] = []
    for sub in ["beta_set", "larger_mols"]:
        d = _MOL_LIB / sub
        if d.is_dir():
            for f in sorted(d.glob("*.mol")):
                name = f.stem
                if not pattern or pattern.lower() in name.lower():
                    mols.append(f"{sub}/{name}")
    if not mols:
        return f"No molecules found matching {pattern!r}" if pattern else "Molecule library is empty"
    return "\n".join(mols)


@server.tool()
def get_molecule(name: str) -> str:
    """Get a molecule geometry from the local library.

    Args:
        name: Molecule name (e.g. "CH3OH", "naphthalene"). Searches beta_set and larger_mols.
    """
    found = _find_molecule(name)
    if found:
        return f"File: {found}\n\n{found.read_text()}"
    return f"Molecule {name!r} not found in library. Use list_molecules() to see available molecules."


@server.tool()
def fetch_pubchem_geometry(name: str) -> str:
    """Fetch a molecular geometry from PubChem by name or formula.

    Args:
        name: Molecule name or formula (e.g. "water", "SO2", "ethanol").
    """
    from gecko.workflow.geometry import fetch_geometry

    mol = fetch_geometry(name)
    lines = [f"Formula: {mol.get_molecular_formula()} ({len(mol.symbols)} atoms)"]
    lines.append(f"Coordinates (Bohr):")
    import numpy as np
    geom = np.array(mol.geometry).reshape(-1, 3)
    for sym, xyz in zip(mol.symbols, geom):
        lines.append(f"  {sym:2s} {xyz[0]:14.8f} {xyz[1]:14.8f} {xyz[2]:14.8f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools — Calculation Loading & Analysis
# ---------------------------------------------------------------------------


@server.tool()
def load_calculation(path: str) -> str:
    """Load a completed MADNESS or DALTON calculation and return a summary.

    Args:
        path: Path to the calculation directory.
    """
    import gecko

    calc = gecko.load_calc(path)
    lines = [f"Code: {calc.code}", f"Root: {calc.root}"]

    if calc.molecule:
        lines.append(f"Molecule: {calc.molecule.get_molecular_formula()} ({len(calc.molecule.symbols)} atoms)")

    if calc.basis:
        lines.append(f"Basis: {calc.basis}")

    if calc.meta.get("ground_state_energy") is not None:
        lines.append(f"Ground state energy: {calc.meta['ground_state_energy']:.10f} Hartree")

    if calc.meta.get("method"):
        lines.append(f"Method: {calc.meta['method']}")

    lines.append(f"Data keys: {', '.join(sorted(calc.data.keys()))}")

    # Summarize available response properties
    if "alpha" in calc.data:
        alpha = calc.data["alpha"]
        if isinstance(alpha, dict) and "omega" in alpha:
            lines.append(f"Alpha: {len(alpha['omega'])} frequencies")

    if "beta" in calc.data:
        beta = calc.data["beta"]
        if isinstance(beta, dict) and "values" in beta:
            lines.append(f"Beta: tensor available")

    if "timings" in calc.data:
        lines.append("Timing data: available")

    return "\n".join(lines)


@server.tool()
def extract_alpha(path: str) -> str:
    """Extract linear polarizability (alpha) data from a calculation.

    Args:
        path: Path to the calculation directory.
    """
    import gecko
    from gecko.tables.builder import TableBuilder

    calc = gecko.load_calc(path)
    builder = TableBuilder([calc])
    try:
        df = builder.build_alpha()
        return df.to_string()
    except Exception as exc:
        return f"Could not extract alpha: {exc}"


@server.tool()
def extract_beta(path: str) -> str:
    """Extract hyperpolarizability (beta) data from a calculation.

    Args:
        path: Path to the calculation directory.
    """
    import gecko
    from gecko.tables.builder import TableBuilder

    calc = gecko.load_calc(path)
    builder = TableBuilder([calc])
    try:
        df = builder.build_beta()
        return df.to_string()
    except Exception as exc:
        return f"Could not extract beta: {exc}"


@server.tool()
def extract_timing(path: str) -> str:
    """Extract timing summary from a calculation.

    Args:
        path: Path to the calculation directory.
    """
    import gecko
    from gecko.tables.builder import TableBuilder

    calc = gecko.load_calc(path)
    builder = TableBuilder([calc])
    try:
        df = builder.build_timing_summary()
        return df.to_string()
    except Exception as exc:
        return f"Could not extract timing: {exc}"


@server.tool()
def compare_calculations(paths: list[str], property: str = "energy") -> str:
    """Compare multiple calculations by extracting a property into a table.

    Args:
        paths: List of calculation directory paths.
        property: What to compare: "energy", "alpha", "beta", "timing".
    """
    import gecko
    from gecko.tables.builder import TableBuilder

    calcs = [gecko.load_calc(p) for p in paths]
    builder = TableBuilder(calcs)

    try:
        if property == "energy":
            df = builder.build_energy()
        elif property == "alpha":
            df = builder.build_alpha()
        elif property == "beta":
            df = builder.build_beta()
        elif property == "timing":
            df = builder.build_timing_summary()
        else:
            return f"Unknown property {property!r}. Valid: energy, alpha, beta, timing"
        return df.to_string()
    except Exception as exc:
        return f"Error comparing {property}: {exc}"


# ---------------------------------------------------------------------------
# Tools — Workflow (calc init)
# ---------------------------------------------------------------------------


@server.tool()
def generate_calc_inputs(
    molecule: str,
    property: str = "alpha",
    xc: str = "hf",
    frequencies: str = "0.0",
    output_dir: str = "",
    geom_file: str = "",
    code: str = "madness",
) -> str:
    """Generate MADNESS (and optionally DALTON) input files for a molecule.

    Uses gecko calc init under the hood. Geometry is loaded from the molecule
    library first, then PubChem as fallback.

    Args:
        molecule: Molecule name or formula (e.g. "CH3OH", "water").
        property: Property to compute: "alpha", "beta", or "raman".
        xc: Exchange-correlation functional (e.g. "hf", "b3lyp").
        frequencies: Comma-separated frequencies in Hartree (e.g. "0.0,0.05").
        output_dir: Output directory. Default: $GECKO_SCRATCH/<molecule>.
        geom_file: Optional path to .xyz/.mol file (overrides library/PubChem).
        code: "madness", "dalton", or "both".
    """
    from gecko.workflow.geometry import fetch_geometry, load_geometry_from_file
    from gecko.workflow.writers import generate_calc_dir

    # Resolve geometry
    if geom_file:
        mol = load_geometry_from_file(Path(geom_file))
    else:
        found = _find_molecule(molecule)
        if found:
            mol = load_geometry_from_file(found)
        else:
            mol = fetch_geometry(molecule)

    codes = ["madness", "dalton"] if code == "both" else [code]
    freqs = [float(f.strip()) for f in frequencies.split(",")]
    out = Path(output_dir) if output_dir else _SCRATCH / molecule

    paths = generate_calc_dir(
        molecule=mol,
        mol_name=molecule,
        property=property,
        codes=codes,
        basis_sets=["aug-cc-pVDZ"],
        frequencies=freqs,
        xc=xc,
        out_dir=out,
    )

    lines = [f"Generated files in {out}:"]
    for c, file_list in paths.items():
        for p in file_list:
            lines.append(f"  [{c}] {p}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Resources — Molecule Library
# ---------------------------------------------------------------------------


@server.resource("gecko://molecules")
def resource_list_molecules() -> str:
    """List all molecules available in the geometry library."""
    mols: list[str] = []
    for sub in ["beta_set", "larger_mols"]:
        d = _MOL_LIB / sub
        if d.is_dir():
            for f in sorted(d.glob("*.mol")):
                mols.append(f"{sub}/{f.stem}")
    return "\n".join(mols) if mols else "No molecules found"


@server.resource("gecko://molecules/{name}")
def resource_get_molecule(name: str) -> str:
    """Get geometry for a specific molecule from the library."""
    found = _find_molecule(name)
    if found:
        return found.read_text()
    return f"Molecule {name!r} not found"


@server.resource("gecko://parameters/{section}")
def resource_parameter_docs(section: str) -> str:
    """Get documentation for all MADNESS parameters in a section."""
    from gecko.workflow.input_model import DFTSection, ResponseSection, MoleculeSection

    cls_map = {"dft": DFTSection, "response": ResponseSection, "molecule": MoleculeSection}
    cls = cls_map.get(section)
    if not cls:
        return f"Unknown section {section!r}. Valid: dft, response, molecule"

    lines: list[str] = []
    for name, info in cls.model_fields.items():
        alias = info.alias or name
        ann = info.annotation
        type_name = getattr(ann, "__name__", str(ann))
        lines.append(f"{alias}  ({type_name}, default={info.default!r})")
        if info.description:
            lines.append(f"    {info.description}")
    return "\n".join(lines)


@server.resource("gecko://schema")
def resource_json_schema() -> str:
    """Full JSON Schema for MADNESS input files."""
    from gecko.workflow.input_model import MadnessInputFile

    return json.dumps(MadnessInputFile.model_json_schema(), indent=2)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@server.prompt()
def setup_calculation(molecule: str, property: str = "alpha", xc: str = "hf") -> str:
    """Guide through setting up a new MADNESS calculation."""
    return (
        f"Help me set up a MADNESS {property} calculation for {molecule} "
        f"using {xc} functional.\n\n"
        f"1. First check the molecule library with list_molecules('{molecule}')\n"
        f"2. If found, use get_molecule('{molecule}') to see the geometry\n"
        f"3. Generate input files with generate_calc_inputs('{molecule}', '{property}', '{xc}')\n"
        f"4. Review the generated input with show_input()\n"
        f"5. Adjust parameters if needed with set_parameter()"
    )


@server.prompt()
def analyze_calculation(path: str) -> str:
    """Guide through analyzing a completed calculation."""
    return (
        f"Help me analyze the calculation at {path}.\n\n"
        f"1. Load and summarize with load_calculation('{path}')\n"
        f"2. Extract response properties with extract_alpha() or extract_beta()\n"
        f"3. Check timing with extract_timing()\n"
        f"4. If there are convergence issues, check the input parameters"
    )


@server.prompt()
def compare_inputs(file1: str, file2: str) -> str:
    """Compare two MADNESS input files and explain the differences."""
    return (
        f"Compare these two MADNESS input files:\n"
        f"  1. {file1}\n"
        f"  2. {file2}\n\n"
        f"Use diff_inputs('{file1}', '{file2}') to get the semantic differences, "
        f"then explain what each difference means for the calculation."
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_molecule(name: str) -> Path | None:
    """Search the molecule library for a .mol file by name."""
    for sub in ["beta_set", "larger_mols"]:
        candidate = _MOL_LIB / sub / f"{name}.mol"
        if candidate.exists():
            return candidate
    # Case-insensitive fallback
    for sub in ["beta_set", "larger_mols"]:
        d = _MOL_LIB / sub
        if d.is_dir():
            for f in d.glob("*.mol"):
                if f.stem.lower() == name.lower():
                    return f
    return None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
