"""fixture_manager — developer validation utilities for gecko.

Subcommands
-----------
validate-all  Compare completed fixture calculations against reference_db.json.
compare       Diff the same fixture run under two different builds.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Tier tolerances
# ---------------------------------------------------------------------------

_TIER_TOLERANCE: dict[str, float] = {
    "low": 1e-2,
    "medium": 1e-4,
    "high": 1e-6,
}

_DEFAULT_FIXTURES_DIR = Path(
    os.environ.get(
        "GECKO_FIXTURES_DIR",
        "/gpfs/projects/rjh/adrian/development/madness-worktrees"
        "/molresponse-feature-next/src/apps/molresponse_v3/tests/fixtures",
    )
)


def _load_reference_db(db_path: Path) -> dict[str, Any]:
    if not db_path.exists():
        raise FileNotFoundError(f"reference_db.json not found: {db_path}")
    return json.loads(db_path.read_text(encoding="utf-8"))


def _load_alpha_from_calc(calc_path: Path) -> dict[str, Any] | None:
    """Load a calc directory and return its alpha data dict, or None."""
    from gecko.core.load import load_calc

    try:
        calc = load_calc(calc_path)
    except Exception as exc:
        return {"_error": str(exc)}
    return calc.data.get("alpha")


def _static_tensor(alpha: dict[str, Any]) -> dict[str, float] | None:
    """Extract static (omega=0) diagonal tensor from alpha data."""
    if not alpha:
        return None
    omega = list(alpha.get("omega", []))
    comps = list(alpha.get("components", []))
    vals = list(alpha.get("values", []))
    if not omega or not comps or not vals:
        return None

    vals_arr = np.asarray(vals, dtype=float)
    # find omega=0 row
    omegas = np.asarray(omega, dtype=float)
    static_idx = int(np.argmin(np.abs(omegas)))

    result = {}
    for j, comp in enumerate(comps):
        result[str(comp).lower()] = float(vals_arr[static_idx, j])
    return result


# ---------------------------------------------------------------------------
# validate-all
# ---------------------------------------------------------------------------


def _validate_all(args: Any) -> int:
    db_path = Path(args.db) if args.db else _DEFAULT_FIXTURES_DIR / "reference_db.json"
    tier: str = args.tier
    tol = _TIER_TOLERANCE.get(tier, 1e-4)

    print(f"Loading reference_db from: {db_path}")
    print(f"Tier: {tier}  |  Tolerance: {tol:.0e}\n")

    ref_db = _load_reference_db(db_path)
    systems: dict[str, Any] = ref_db.get("systems", {})

    if not systems:
        print("No systems found in reference_db.json.")
        return 1

    overall_pass = True
    results: list[dict[str, Any]] = []

    for mol_name, sys_data in systems.items():
        sources: dict[str, Any] = sys_data.get("sources", {})
        ref_alpha_static = sys_data.get("alpha", {}).get("static", {})
        ref_tensor: dict[str, float] = ref_alpha_static.get("tensor", {})

        if not ref_tensor:
            results.append({"system": mol_name, "status": "SKIP", "reason": "no reference alpha"})
            continue

        # Pick first available source that has alpha
        calc_path: Path | None = None
        for src_name, src in sources.items():
            if "alpha" in src.get("properties_available", []):
                p = Path(src["path"])
                if p.exists():
                    calc_path = p
                    break

        if calc_path is None:
            results.append({"system": mol_name, "status": "SKIP", "reason": "no accessible calc path"})
            continue

        alpha = _load_alpha_from_calc(calc_path)

        if alpha is None:
            results.append({"system": mol_name, "status": "FAIL", "reason": "alpha not found in calc"})
            overall_pass = False
            continue

        if "_error" in alpha:
            results.append({"system": mol_name, "status": "FAIL", "reason": alpha["_error"]})
            overall_pass = False
            continue

        computed = _static_tensor(alpha)
        if not computed:
            results.append({"system": mol_name, "status": "FAIL", "reason": "could not extract static tensor"})
            overall_pass = False
            continue

        component_results: list[dict[str, Any]] = []
        system_pass = True
        for comp, ref_val in ref_tensor.items():
            calc_val = computed.get(comp)
            if calc_val is None:
                component_results.append({"comp": comp, "status": "FAIL", "reason": "component missing"})
                system_pass = False
                continue
            diff = abs(calc_val - ref_val)
            passed = diff <= tol
            if not passed:
                system_pass = False
            component_results.append({
                "comp": comp,
                "ref": ref_val,
                "calc": calc_val,
                "diff": diff,
                "tol": tol,
                "status": "PASS" if passed else "FAIL",
            })

        results.append({
            "system": mol_name,
            "status": "PASS" if system_pass else "FAIL",
            "components": component_results,
            "source": str(calc_path),
        })
        if not system_pass:
            overall_pass = False

    # Print report
    width = 60
    print("=" * width)
    print(f"{'SYSTEM':<12}  {'STATUS':<6}  DETAILS")
    print("=" * width)
    for r in results:
        status = r["status"]
        marker = "✓" if status == "PASS" else ("−" if status == "SKIP" else "✗")
        print(f"  {marker}  {r['system']:<10}  {status}")
        if "components" in r:
            for c in r["components"]:
                s = c["status"]
                m = "✓" if s == "PASS" else "✗"
                if "reason" in c:
                    print(f"       {m}  {c['comp']:<4}  {c['reason']}")
                else:
                    print(
                        f"       {m}  {c['comp']:<4}  "
                        f"ref={c['ref']:.6f}  calc={c['calc']:.6f}  "
                        f"diff={c['diff']:.2e}  tol={c['tol']:.0e}"
                    )
        elif "reason" in r:
            print(f"         reason: {r['reason']}")
    print("=" * width)
    print(f"\nOverall: {'PASS' if overall_pass else 'FAIL'}")
    return 0 if overall_pass else 1


# ---------------------------------------------------------------------------
# compare
# ---------------------------------------------------------------------------


def _compare(args: Any) -> int:
    build1 = Path(args.build1)
    build2 = Path(args.build2)
    prop: str = args.property
    tier: str = args.tier
    tol = _TIER_TOLERANCE.get(tier, 1e-4)

    for p, label in [(build1, "--build1"), (build2, "--build2")]:
        if not p.exists():
            print(f"Error: {label} path does not exist: {p}")
            return 1

    from gecko.core.load import load_calc

    print(f"Loading build1: {build1}")
    calc1 = load_calc(build1)
    print(f"Loading build2: {build2}")
    calc2 = load_calc(build2)

    print(f"\nComparing property: {prop}  |  Tier: {tier}  |  Tolerance: {tol:.0e}\n")

    if prop == "alpha":
        d1 = calc1.data.get("alpha")
        d2 = calc2.data.get("alpha")
        _compare_tensor_property(d1, d2, "alpha", tol)
    elif prop == "beta":
        d1 = calc1.data.get("beta")
        d2 = calc2.data.get("beta")
        _compare_tensor_property(d1, d2, "beta", tol)
    elif prop == "energy":
        _compare_energy(calc1, calc2, tol)
    else:
        print(f"Property '{prop}' comparison not yet supported.")
        return 1

    return 0


def _compare_tensor_property(
    d1: dict[str, Any] | None,
    d2: dict[str, Any] | None,
    label: str,
    tol: float,
) -> None:
    if d1 is None or d2 is None:
        missing = "build1" if d1 is None else "build2"
        print(f"FAIL: {label} not found in {missing}")
        return

    comps1 = list(d1.get("components", []))
    comps2 = list(d2.get("components", []))
    if comps1 != comps2:
        print(f"WARNING: component lists differ\n  build1: {comps1}\n  build2: {comps2}")

    vals1 = np.asarray(d1.get("values", []), dtype=float)
    vals2 = np.asarray(d2.get("values", []), dtype=float)

    if vals1.shape != vals2.shape:
        print(f"FAIL: value array shapes differ: {vals1.shape} vs {vals2.shape}")
        return

    omega1 = np.asarray(d1.get("omega", []), dtype=float).reshape(-1)

    regressions = 0
    width = 70
    print("=" * width)
    print(f"{'COMP':<6}  {'OMEGA':<10}  {'BUILD1':>14}  {'BUILD2':>14}  {'DIFF':>12}  STATUS")
    print("=" * width)
    for i, om in enumerate(omega1 if omega1.ndim == 1 else omega1[:, 0]):
        for j, comp in enumerate(comps1):
            v1 = float(vals1[i, j])
            v2 = float(vals2[i, j])
            diff = abs(v1 - v2)
            passed = diff <= tol
            if not passed:
                regressions += 1
            marker = "✓" if passed else "✗"
            print(
                f"  {marker}  {str(comp):<4}  {float(om):>10.6f}  "
                f"{v1:>14.6f}  {v2:>14.6f}  {diff:>12.2e}  "
                f"{'PASS' if passed else 'FAIL'}"
            )
    print("=" * width)
    print(f"\n{'No regressions found.' if regressions == 0 else f'{regressions} regression(s) found.'}")


def _compare_energy(calc1: Any, calc2: Any, tol: float) -> None:
    from gecko.tables.extractors import extract_energy

    rows1 = extract_energy(calc1)
    rows2 = extract_energy(calc2)

    e1 = rows1[0]["energy"] if rows1 else None
    e2 = rows2[0]["energy"] if rows2 else None

    if e1 is None or e2 is None:
        missing = "build1" if e1 is None else "build2"
        print(f"FAIL: energy not found in {missing}")
        return

    diff = abs(e1 - e2)
    passed = diff <= tol
    marker = "✓" if passed else "✗"
    print(f"  {marker}  energy  build1={e1:.10f}  build2={e2:.10f}  diff={diff:.2e}  tol={tol:.0e}  {'PASS' if passed else 'FAIL'}")
