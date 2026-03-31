"""MADNESS MCP Server — exposes MADNESS build and execution tools.

Wraps the madqc compute engine and MADNESS build system, providing
tools for running calculations, rebuilding targets, and inspecting
MADNESS-side artifacts.

Run standalone::

    python -m gecko.madness_mcp_server

Or configure in .mcp.json for Claude Code integration.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("madness")

# ---------------------------------------------------------------------------
# Paths (configurable via env vars)
# ---------------------------------------------------------------------------

_MADNESS_SRC = Path(
    os.environ.get(
        "MADNESS_SRC",
        "/gpfs/projects/rjh/adrian/development/madness-worktrees/molresponse-feature-next",
    )
)

_MADNESS_BUILD = Path(
    os.environ.get(
        "MADNESS_BUILD",
        "/gpfs/projects/rjh/adrian/development/madness-worktrees/builds/molresponse-feature-next/debug",
    )
)

_MADQC_EXEC = os.environ.get("MADQC_EXEC", str(_MADNESS_BUILD / "src" / "apps" / "madqc_v2" / "madqc"))

_TEST_DIR = _MADNESS_SRC / "src" / "apps" / "madqc_v2"


# ---------------------------------------------------------------------------
# Tools — Build
# ---------------------------------------------------------------------------


@server.tool()
def rebuild(target: str = "") -> str:
    """Rebuild a MADNESS target with ninja.

    Args:
        target: Ninja target name (e.g. "madqc", "molresponse2", "moldft").
                Empty string rebuilds everything.
    """
    cmd = ["ninja"]
    if target:
        cmd.append(target)

    result = subprocess.run(
        cmd,
        cwd=str(_MADNESS_BUILD),
        capture_output=True,
        text=True,
        timeout=600,
    )

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr
    if result.returncode != 0:
        output += f"\n\nBuild FAILED (exit code {result.returncode})"
    else:
        output += "\n\nBuild succeeded."

    # Truncate if very long
    if len(output) > 4000:
        output = output[:2000] + "\n... (truncated) ...\n" + output[-2000:]

    return output


@server.tool()
def build_info() -> str:
    """Show MADNESS build configuration and paths."""
    lines = [
        f"Source:    {_MADNESS_SRC}",
        f"Build:     {_MADNESS_BUILD}",
        f"madqc:     {_MADQC_EXEC}",
    ]

    # Check if madqc binary exists
    madqc = Path(_MADQC_EXEC)
    if madqc.exists():
        import time
        mtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(madqc.stat().st_mtime))
        lines.append(f"madqc modified: {mtime}")
    else:
        lines.append("madqc: NOT FOUND (run rebuild('madqc'))")

    # Check build.ninja
    build_ninja = _MADNESS_BUILD / "build.ninja"
    if build_ninja.exists():
        lines.append(f"build.ninja: exists")
    else:
        lines.append(f"build.ninja: NOT FOUND")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools — Running calculations
# ---------------------------------------------------------------------------


@server.tool()
def run_madqc(input_file: str, wf: str = "response", working_dir: str = "") -> str:
    """Run a MADNESS calculation via madqc.

    This runs the calculation synchronously and returns the output.
    For large calculations, consider using run_madqc_background instead.

    Args:
        input_file: Path to the .in input file.
        wf: Workflow type: "response" (default), "moldft", etc.
        working_dir: Working directory for the run. Default: directory containing the input file.
    """
    input_path = Path(input_file).resolve()
    cwd = working_dir if working_dir else str(input_path.parent)

    result = subprocess.run(
        [_MADQC_EXEC, f"--wf={wf}", str(input_path)],
        cwd=cwd,
        capture_output=True,
        text=True,
        timeout=3600,
    )

    output = result.stdout
    if result.stderr:
        output += "\n--- stderr ---\n" + result.stderr[-1000:]

    # Truncate long output, keeping head and tail
    if len(output) > 6000:
        output = output[:3000] + "\n\n... (truncated) ...\n\n" + output[-3000:]

    if result.returncode != 0:
        output += f"\n\nExited with code {result.returncode}"

    return output


@server.tool()
def check_madqc_output(calc_dir: str) -> str:
    """Check the output files from a madqc run in a directory.

    Looks for calc_info.json, response_metadata.json, and .out files.

    Args:
        calc_dir: Path to the calculation directory.
    """
    d = Path(calc_dir)
    if not d.is_dir():
        return f"Not a directory: {calc_dir}"

    lines = [f"Directory: {d}"]

    # Look for key output files
    for pattern in ["*.calc_info.json", "response_metadata.json", "*.out", "*.in", "*.restartdata"]:
        matches = list(d.glob(pattern))
        if matches:
            for m in sorted(matches):
                size = m.stat().st_size
                lines.append(f"  {m.name} ({size:,} bytes)")

    # If calc_info.json exists, summarize it
    calc_infos = list(d.glob("*.calc_info.json"))
    if calc_infos:
        try:
            data = json.loads(calc_infos[0].read_text())
            if "return_energy" in data:
                lines.append(f"\n  Ground state energy: {data['return_energy']:.10f}")
            if "wall_time" in data:
                lines.append(f"  Wall time: {data['wall_time']:.1f}s")
        except Exception:
            pass

    # Check response_metadata.json
    meta = d / "response_metadata.json"
    if meta.exists():
        try:
            data = json.loads(meta.read_text())
            if "state_points" in data:
                lines.append(f"\n  State points: {len(data['state_points'])}")
            if "timing" in data:
                lines.append(f"  Timing data available")
        except Exception:
            pass

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools — Test fixtures and examples
# ---------------------------------------------------------------------------


@server.tool()
def list_test_inputs() -> str:
    """List MADNESS test input files (useful as examples/templates)."""
    if not _TEST_DIR.is_dir():
        return f"Test directory not found: {_TEST_DIR}"

    lines = [f"Test inputs in {_TEST_DIR}:"]
    for f in sorted(_TEST_DIR.glob("*.in")):
        lines.append(f"  {f.name}")
    return "\n".join(lines)


@server.tool()
def read_test_input(name: str) -> str:
    """Read a MADNESS test input file by name.

    Args:
        name: Filename (e.g. "test_molresponse_h2o_alpha_beta_z.in") or stem without .in.
    """
    if not name.endswith(".in"):
        name = name + ".in"
    path = _TEST_DIR / name
    if not path.exists():
        return f"Test input not found: {path}\nUse list_test_inputs() to see available files."
    return path.read_text()


# ---------------------------------------------------------------------------
# Tools — Source code inspection
# ---------------------------------------------------------------------------


@server.tool()
def search_madness_source(pattern: str, file_glob: str = "*.hpp,*.h,*.cpp") -> str:
    """Search MADNESS source code for a pattern using grep.

    Args:
        pattern: Search pattern (case-insensitive).
        file_glob: Comma-separated file extensions to search (default: "*.hpp,*.h,*.cpp").
    """
    globs = [g.strip() for g in file_glob.split(",")]
    include_args: list[str] = []
    for g in globs:
        include_args.extend(["--include", g])

    result = subprocess.run(
        ["grep", "-r", "-i", "-n", "--max-count=50", pattern] + include_args + [str(_MADNESS_SRC / "src")],
        capture_output=True,
        text=True,
        timeout=30,
    )

    output = result.stdout
    if not output:
        return f"No matches for {pattern!r}"

    # Make paths relative
    src_prefix = str(_MADNESS_SRC) + "/"
    output = output.replace(src_prefix, "")

    # Truncate if very long
    lines = output.splitlines()
    if len(lines) > 50:
        output = "\n".join(lines[:50]) + f"\n... ({len(lines)} total matches)"

    return output


@server.tool()
def read_madness_file(path: str, start_line: int = 1, num_lines: int = 100) -> str:
    """Read a MADNESS source file (relative to source root).

    Args:
        path: Relative path from MADNESS source root (e.g. "src/madness/chem/ResponseParameters.hpp").
        start_line: Starting line number (1-indexed).
        num_lines: Number of lines to read (default 100).
    """
    full_path = _MADNESS_SRC / path
    if not full_path.exists():
        return f"File not found: {full_path}"

    lines = full_path.read_text().splitlines()
    end = min(start_line - 1 + num_lines, len(lines))
    selected = lines[start_line - 1 : end]

    header = f"File: {path} (lines {start_line}-{end} of {len(lines)})\n"
    return header + "\n".join(f"{i:5d}  {line}" for i, line in enumerate(selected, start=start_line))


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@server.resource("madness://build-info")
def resource_build_info() -> str:
    """MADNESS build configuration and paths."""
    return build_info()


@server.resource("madness://test-inputs")
def resource_test_inputs() -> str:
    """List of available MADNESS test input files."""
    return list_test_inputs()


@server.resource("madness://parameters/{section}")
def resource_parameters(section: str) -> str:
    """Documentation for MADNESS parameters in a section (dft/response/molecule)."""
    from gecko.workflow.input_model import DFTSection, ResponseSection, MoleculeSection

    cls_map = {"dft": DFTSection, "response": ResponseSection, "molecule": MoleculeSection}
    cls = cls_map.get(section)
    if not cls:
        return f"Unknown section: {section}"

    lines: list[str] = []
    for name, info in cls.model_fields.items():
        alias = info.alias or name
        ann = info.annotation
        type_name = getattr(ann, "__name__", str(ann))
        lines.append(f"{alias}  ({type_name}, default={info.default!r})")
        if info.description:
            lines.append(f"    {info.description}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@server.prompt()
def debug_calculation(input_file: str) -> str:
    """Help debug a MADNESS calculation that isn't converging."""
    return (
        f"Help me debug a MADNESS calculation.\n\n"
        f"Input file: {input_file}\n\n"
        f"Steps:\n"
        f"1. Read the input with read_test_input() or show_input() from gecko server\n"
        f"2. Check if parameters look reasonable (convergence thresholds, box size, etc.)\n"
        f"3. If there's output, check it with check_madqc_output()\n"
        f"4. Look at the response parameters — is dconv tight enough? Is protocol sensible?\n"
        f"5. Suggest parameter changes to improve convergence"
    )


@server.prompt()
def explore_parameter(param_name: str) -> str:
    """Understand what a MADNESS parameter does by searching the source code."""
    return (
        f"Help me understand the MADNESS parameter '{param_name}'.\n\n"
        f"1. Search the source for its definition with search_madness_source('{param_name}')\n"
        f"2. Check the parameter documentation with resource_parameters()\n"
        f"3. Find where it's used in the calculation pipeline\n"
        f"4. Explain what it does and when to change it"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    server.run()


if __name__ == "__main__":
    main()
