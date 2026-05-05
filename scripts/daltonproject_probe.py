"""Probe daltonproject.dalton for all 9 fixture molecules.

Tests whether daltonproject.dalton can reproduce polarizability (HF, static)
for each fixture molecule. Runs two passes: cc-pVDZ then d-aug-cc-pVDZ.
Compares against reference_db.json where reference values exist.

Usage
-----
    python scripts/daltonproject_probe.py
    python scripts/daltonproject_probe.py --basis dz       # cc-pVDZ only
    python scripts/daltonproject_probe.py --basis daug     # d-aug-cc-pVDZ only
    python scripts/daltonproject_probe.py --mol H2O LiH    # subset

Open-shell (Li, NO, OH) are handled via ROHF: multiplicity=2 +
method.scf_occupation(closed_shells=[n], open_shells=[1]).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Molecule definitions
# ---------------------------------------------------------------------------

_mol_lib = os.environ.get("GECKO_MOL_LIB")
if not _mol_lib:
    raise EnvironmentError("Set GECKO_MOL_LIB to the molecules directory before running this script.")
MOL_LIB = Path(_mol_lib)

_fixtures_dir = os.environ.get("GECKO_FIXTURES_DIR")
if not _fixtures_dir:
    raise EnvironmentError("Set GECKO_FIXTURES_DIR to the fixtures directory before running this script.")
FIXTURES_DIR = Path(_fixtures_dir)

REFERENCE_DB = FIXTURES_DIR / "reference_db.json"


@dataclass
class MolSpec:
    """Everything daltonproject needs to build a Molecule."""
    name: str
    # atoms string: "ELEM x y z; ELEM x y z ..." (Angstrom)
    atoms: str
    charge: int = 0
    multiplicity: int = 1
    # ROHF occupations for open-shell; None for closed-shell
    closed_shells: Optional[int] = None
    open_shells: Optional[int] = None


# Geometries taken from the gecko molecule library (.mol files).
# Coordinates in Angstrom.
MOLECULES: list[MolSpec] = [
    # ---------- closed-shell ----------
    MolSpec(
        name="He",
        atoms="He 0.0 0.0 0.0",
    ),
    MolSpec(
        name="H2",
        atoms="H 0.0 0.0 0.0; H 0.0 0.0 0.7414",
    ),
    MolSpec(
        name="LiH",
        atoms="Li 0.0 0.0 0.0; H 0.0 0.0 1.5949",
    ),
    MolSpec(
        name="H2O",
        atoms="O 0.0 0.0 0.11730; H 0.0 0.75720 -0.46920; H 0.0 -0.75720 -0.46920",
    ),
    MolSpec(
        name="C6H6",
        atoms=(
            "C  0.000000  1.395248  0.0;"
            "C  1.208320  0.697624  0.0;"
            "C  1.208320 -0.697624  0.0;"
            "C  0.000000 -1.395248  0.0;"
            "C -1.208320 -0.697624  0.0;"
            "C -1.208320  0.697624  0.0;"
            "H  0.000000  2.482360  0.0;"
            "H  2.149787  1.241180  0.0;"
            "H  2.149787 -1.241180  0.0;"
            "H  0.000000 -2.482360  0.0;"
            "H -2.149787 -1.241180  0.0;"
            "H -2.149787  1.241180  0.0"
        ),
    ),
    MolSpec(
        name="naphthalene",
        atoms=(
            "C  0.653641  0.296519  0.0;"
            "C -0.653642 -0.296519  0.0;"
            "C -1.793838  0.554476  0.0;"
            "C -1.653077  1.925923  0.0;"
            "C -0.361331  2.511976  0.0;"
            "C  0.763663  1.714906  0.0;"
            "C  1.793837 -0.554476  0.0;"
            "C  1.653077 -1.925923  0.0;"
            "C  0.361331 -2.511976  0.0;"
            "C -0.763663 -1.714906  0.0;"
            "H -2.788434  0.099968  0.0;"
            "H -2.537508  2.568040  0.0;"
            "H -0.262853  3.600634  0.0;"
            "H  1.760668  2.163952  0.0;"
            "H  2.788433 -0.099966  0.0;"
            "H  2.537509 -2.568039  0.0;"
            "H  0.262854 -3.600633  0.0;"
            "H -1.760668 -2.163956  0.0"
        ),
    ),
    # ---------- open-shell (doublet, ROHF) ----------
    # Li: 1s²2s¹ → 2 electrons in 1 closed shell, 1 electron in 1 open shell
    MolSpec(
        name="Li",
        atoms="Li 0.0 0.0 0.0",
        multiplicity=2,
        closed_shells=1,
        open_shells=1,
    ),
    # NO: 15 electrons → 7 closed, 1 open (π*)
    MolSpec(
        name="NO",
        atoms="N 0.0 0.0 0.0; O 0.0 0.0 1.1508",
        multiplicity=2,
        closed_shells=7,
        open_shells=1,
    ),
    # OH: 9 electrons → 4 closed, 1 open
    MolSpec(
        name="OH",
        atoms="O 0.0 0.0 0.0; H 0.0 0.0 0.9696",
        multiplicity=2,
        closed_shells=4,
        open_shells=1,
    ),
]


# ---------------------------------------------------------------------------
# Reference values from reference_db.json
# ---------------------------------------------------------------------------

def load_reference_values() -> dict[str, dict]:
    """Return {mol_name: {"xx": float, "yy": float, "zz": float, "iso": float}}."""
    if not REFERENCE_DB.exists():
        return {}
    db = json.loads(REFERENCE_DB.read_text())
    out = {}
    for mol_name, sys_data in db.get("systems", {}).items():
        tensor = sys_data.get("alpha", {}).get("static", {}).get("tensor")
        iso = sys_data.get("alpha", {}).get("static", {}).get("isotropic")
        if tensor:
            out[mol_name] = {
                "xx": tensor.get("xx"),
                "yy": tensor.get("yy"),
                "zz": tensor.get("zz"),
                "iso": iso,
            }
    return out


# ---------------------------------------------------------------------------
# Single calculation
# ---------------------------------------------------------------------------

@dataclass
class CalcResult:
    mol: str
    basis: str
    alpha_xx: Optional[float] = None
    alpha_yy: Optional[float] = None
    alpha_zz: Optional[float] = None
    alpha_iso: Optional[float] = None
    error: Optional[str] = None


def run_one(spec: MolSpec, basis_name: str, scratch_dir: Path, force: bool = False) -> CalcResult:
    """Run a single HF polarizability calculation via daltonproject."""
    result = CalcResult(mol=spec.name, basis=basis_name)

    try:
        import daltonproject as dp
    except ImportError:
        result.error = "daltonproject not installed — run: pip install git+https://gitlab.com/daltonproject/daltonproject"
        return result

    try:
        mol = dp.Molecule(atoms=spec.atoms, charge=spec.charge, multiplicity=spec.multiplicity)
        basis = dp.Basis(basis=basis_name)
        method = dp.QCMethod("HF")

        if spec.closed_shells is not None:
            method.scf_occupation(
                closed_shells=[spec.closed_shells],
                open_shells=[spec.open_shells],
            )

        prop = dp.Property(polarizabilities=True)

        calc_dir = scratch_dir / spec.name / basis_name.replace("-", "_").replace("/", "_")
        calc_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{spec.name}_{basis_name}"

        # work_dir must match the directory of the input files; dalton resolves
        # -mol/-dal relative to work_dir, so we pass just the stem as filename.
        settings = dp.ComputeSettings(work_dir=str(calc_dir))

        out = dp.dalton.compute(
            mol,
            basis,
            method,
            prop,
            filename=stem,
            compute_settings=settings,
            force_recompute=force,
        )

        pol = out.polarizabilities
        # static: omega=0, index 0
        alpha = pol.values[0]
        result.alpha_xx = float(alpha[0, 0])
        result.alpha_yy = float(alpha[1, 1])
        result.alpha_zz = float(alpha[2, 2])
        result.alpha_iso = float(np.trace(alpha) / 3.0)

    except Exception as exc:
        result.error = str(exc)

    return result


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def print_report(
    results: list[CalcResult],
    refs: dict[str, dict],
    tol: float = 1e-2,
) -> None:
    w = 100
    print()
    print("=" * w)
    print(f"{'MOL':<14} {'BASIS':<18} {'alpha_xx':>10} {'alpha_yy':>10} {'alpha_zz':>10} {'alpha_iso':>10}  STATUS / NOTE")
    print("=" * w)

    for r in results:
        if r.error:
            print(f"  {'✗':<2} {r.mol:<12} {r.basis:<18}  ERROR: {r.error}")
            continue

        ref = refs.get(r.mol)
        if ref and ref.get("iso") is not None:
            diff = abs(r.alpha_iso - ref["iso"])
            status = "PASS" if diff <= tol else f"FAIL (diff={diff:.2e}, ref={ref['iso']:.4f})"
        else:
            status = "(no reference)"

        marker = "✓" if "PASS" in status else ("?" if "no reference" in status else "✗")
        print(
            f"  {marker}  {r.mol:<12} {r.basis:<18}"
            f"  {r.alpha_xx:>10.4f} {r.alpha_yy:>10.4f} {r.alpha_zz:>10.4f}"
            f"  {r.alpha_iso:>10.4f}  {status}"
        )

    print("=" * w)

    # Summary of open-shell systems
    open_shell_names = {"Li", "NO", "OH"}
    open_results = [r for r in results if r.mol in open_shell_names]
    if open_results:
        print()
        print("Open-shell (ROHF) systems:")
        for r in open_results:
            if r.error:
                print(f"  {r.mol:<6} {r.basis:<18}  FAILED: {r.error}")
            else:
                print(f"  {r.mol:<6} {r.basis:<18}  OK — iso={r.alpha_iso:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--basis", choices=["dz", "daug", "both"], default="both",
        help="Which basis to run: dz=cc-pVDZ, daug=d-aug-cc-pVDZ, both (default)",
    )
    parser.add_argument(
        "--mol", nargs="+", default=None,
        help="Subset of molecules to run (default: all 9)",
    )
    parser.add_argument(
        "--scratch", default=None,
        help="Scratch directory for daltonproject output (default: $GECKO_SCRATCH or ./dalton_probe_out)",
    )
    parser.add_argument(
        "--tol", type=float, default=1e-2,
        help="Tolerance for PASS/FAIL comparison vs reference (default: 1e-2 = low tier)",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Force recompute even if output already exists",
    )
    args = parser.parse_args()

    scratch = Path(
        args.scratch
        or os.environ.get("GECKO_SCRATCH", "")
        or "./dalton_probe_out"
    )
    scratch.mkdir(parents=True, exist_ok=True)

    basis_map = {
        "dz":   ["cc-pVDZ"],
        "daug": ["d-aug-cc-pVDZ"],
        "both": ["cc-pVDZ", "d-aug-cc-pVDZ"],
    }
    bases = basis_map[args.basis]

    mols = MOLECULES
    if args.mol:
        requested = set(args.mol)
        mols = [m for m in MOLECULES if m.name in requested]
        missing = requested - {m.name for m in mols}
        if missing:
            print(f"Warning: unknown molecule(s): {missing}", file=sys.stderr)

    refs = load_reference_values()
    print(f"Loaded {len(refs)} reference systems from reference_db.json")
    print(f"Scratch dir: {scratch}")
    print(f"Molecules:   {[m.name for m in mols]}")
    print(f"Bases:       {bases}")
    print()

    all_results: list[CalcResult] = []
    for basis_name in bases:
        print(f"--- Running basis: {basis_name} ---")
        for spec in mols:
            print(f"  {spec.name} ... ", end="", flush=True)
            r = run_one(spec, basis_name, scratch, force=args.force)
            if r.error:
                print(f"ERROR: {r.error}")
            else:
                print(f"iso={r.alpha_iso:.4f}")
            all_results.append(r)

    print_report(all_results, refs, tol=args.tol)

    errors = [r for r in all_results if r.error]
    if errors:
        print(f"\n{len(errors)} calculation(s) failed.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
