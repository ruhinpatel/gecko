from __future__ import annotations

from pathlib import Path
from typing import Any

from gecko.core.model import Calculation
from gecko.plugins.madness.detect import can_load as madness_can_load
from gecko.plugins.dalton.detect import can_load as dalton_can_load, DaltonCandidate

from gecko.plugins.madness.loader import load as load_madness
from gecko.plugins.dalton.loader import load as load_dalton


def _maybe_attach_calc_info(calc: Calculation) -> None:
    """Attach optional calc_info.json payload without affecting parsing."""
    info_path = calc.root / "calc_info.json"
    if not info_path.exists():
        return
    try:
        import json
        calc.data.setdefault("calc_info", json.loads(info_path.read_text(encoding="utf-8")))
    except Exception as exc:
        calc.meta.setdefault("warnings", []).append(
            f"Failed to read calc_info.json: {info_path} ({type(exc).__name__}: {exc})"
        )


def load_calc(
    path: str | Path,
    *,
    mol_resolver: Any | None = None,
    run: DaltonCandidate | None = None,
) -> Calculation:
    """
    Load and parse a calculation directory.

    Step 2 behavior:
    - Detect whether the directory is MADNESS or DALTON
    - Delegate to the appropriate plugin loader, which parses artifacts
    """
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        if root.suffix.lower() == ".out":
            calc = load_dalton(root.parent, output_file=root, run_id=root.name)
            _maybe_attach_calc_info(calc)
            return calc
        raise ValueError(f"Expected a directory, got a file: {root}")

    if run is not None:
        code = run.get("code")
        if code == "dalton":
            artifacts = run.get("artifacts") or {}
            meta = run.get("meta") or {}
            out_file = artifacts.get("out") or artifacts.get("output")
            run_id = meta.get("out_file")
            calc = load_dalton(root, output_file=out_file, run_id=run_id, meta=meta)
        else:
            raise ValueError(f"Unsupported run code: {code}")
    else:
        if madness_can_load(root):
            calc = load_madness(root)
        elif dalton_can_load(root):
            calc = load_dalton(root)
        else:
            raise ValueError(
                "Could not detect calculation type (madness/dalton) from directory. "
                f"Path: {root}"
            )

    _maybe_attach_calc_info(calc)

    if calc.molecule is None and calc.data.get("molecule") is not None:
        calc.molecule = calc.data.get("molecule")

    if calc.molecule is None and mol_resolver is not None:
        res = mol_resolver.resolve(calc)
        calc.molecule = res.molecule
        calc.meta["mol_source"] = res.source
        calc.meta["mol_path"] = str(res.path) if res.path else None
        calc.meta["mol_label"] = res.label
        if res.source != "missing":
            calc.meta.setdefault("molecule_source", f"resolver:{res.source}")
        if res.path is not None:
            calc.meta.setdefault("molecule_path", str(res.path))
    elif calc.molecule is None:
        calc.meta.setdefault("mol_source", "missing")

    return calc
