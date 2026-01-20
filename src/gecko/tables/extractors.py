from __future__ import annotations

from typing import Any

import numpy as np

from gecko.core.model import Calculation


def make_envelope(calc: Calculation) -> dict[str, Any]:
    return {
        "calc_id": calc.meta.get("calc_id"),
        "geom_id": calc.meta.get("geom_id"),
        "mol_id": calc.meta.get("mol_id"),
        "molecule_id": calc.meta.get("molecule_id"),
        "label": calc.meta.get("label"),
        "code": calc.code,
        "root": str(calc.root),
        "basis": calc.meta.get("basis"),
        "method": calc.meta.get("method"),
    }


def _require_geom(env: dict[str, Any]) -> bool:
    return env.get("geom_id") is not None


def extract_beta(calc: Calculation) -> list[dict[str, Any]]:
    beta = calc.data.get("beta") or {}
    if not beta:
        return []
    if not all(k in beta for k in ("omega", "components", "values")):
        return []

    env = make_envelope(calc)

    omega = np.asarray(beta["omega"], dtype=float)
    comps = list(beta["components"])
    vals = np.asarray(beta["values"], dtype=float)

    rows: list[dict[str, Any]] = []
    for i in range(vals.shape[0]):
        omegaA, omegaB, omegaC = map(float, omega[i])
        for j, ijk in enumerate(comps):
            rows.append(
                {
                    **env,
                    "omegaA": omegaA,
                    "omegaB": omegaB,
                    "omegaC": omegaC,
                    "ijk": str(ijk).lower(),
                    "value": float(vals[i, j]),
                }
            )
    return rows


def extract_alpha(calc: Calculation) -> list[dict[str, Any]]:
    alpha = calc.data.get("alpha")
    if not alpha:
        return []

    env = make_envelope(calc)
    if not _require_geom(env):
        return []

    omega = np.asarray(alpha.get("omega", []), dtype=float).reshape(-1)
    comps = list(alpha.get("components", []))
    vals = np.asarray(alpha.get("values", []), dtype=float)

    rows: list[dict[str, Any]] = []
    for i, om in enumerate(omega):
        for j, ij in enumerate(comps):
            rows.append({**env, "omega": float(om), "ij": str(ij).lower(), "value": float(vals[i, j])})
    return rows


def _find_task(raw_json: dict[str, Any], task_type: str) -> dict[str, Any] | None:
    tasks = raw_json.get("tasks", [])
    if isinstance(tasks, list):
        for t in tasks:
            if isinstance(t, dict) and t.get("type") == task_type:
                return t
    return None


def extract_energy(calc: Calculation) -> list[dict[str, Any]]:
    env = make_envelope(calc)
    if not _require_geom(env):
        return []

    energy = calc.meta.get("ground_state_energy")
    if energy is None:
        raw = calc.data.get("raw_json")
        if isinstance(raw, dict):
            scf = _find_task(raw, "scf")
            if isinstance(scf, dict):
                energy = scf.get("energy")

    if energy is None:
        return []

    return [{**env, "energy": float(energy)}]


def extract_dipole(calc: Calculation) -> list[dict[str, Any]]:
    env = make_envelope(calc)
    if not _require_geom(env):
        return []

    raw = calc.data.get("raw_json")
    if not isinstance(raw, dict):
        return []

    scf = _find_task(raw, "scf")
    if not isinstance(scf, dict):
        return []

    dip = scf.get("dipole", {})
    vals = dip.get("vals") if isinstance(dip, dict) else None
    if not isinstance(vals, list) or len(vals) < 3:
        return []

    rows = []
    for comp, value in zip(["x", "y", "z"], vals, strict=False):
        rows.append({**env, "i": comp, "value": float(value)})
    return rows
