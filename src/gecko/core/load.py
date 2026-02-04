from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from gecko.core.model import Calculation
from gecko.plugins.madness.detect import can_load as madness_can_load
from gecko.plugins.dalton.detect import can_load as dalton_can_load

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


def _maybe_load_molecule_from_calc_dir(calc: Calculation) -> Optional[Any]:
    if calc.molecule is not None:
        return calc.molecule

    root = calc.root
    preferred = root / "molecule.mol"
    candidates: list[Path] = []
    if preferred.exists():
        candidates.append(preferred)
    candidates.extend(sorted(p for p in root.glob("*.mol") if p != preferred))

    if not candidates:
        return None

    from gecko.mol.io import read_mol

    for path in candidates:
        try:
            mol = read_mol(path)
            calc.meta.setdefault("molecule_source", "calc_dir")
            calc.meta.setdefault("molecule_path", str(path))
            return mol
        except Exception as exc:
            calc.meta.setdefault("warnings", []).append(
                f"Failed to read molecule .mol file: {path} ({type(exc).__name__}: {exc})"
            )
    return None


def _finalize_calc(calc: Calculation) -> Calculation:
    _maybe_attach_calc_info(calc)

    if calc.molecule is None and calc.data.get("molecule") is not None:
        calc.molecule = calc.data.get("molecule")

    if calc.molecule is None:
        calc.molecule = _maybe_load_molecule_from_calc_dir(calc)

    if calc.molecule is None:
        calc.meta.setdefault("mol_source", "missing")
    else:
        calc.meta.setdefault("mol_source", "embedded")

    if calc.molecule is not None and calc.meta.get("molecule_id") is None:
        from gecko.ids import geom_id_from_molecule

        calc.meta["molecule_id"] = geom_id_from_molecule(calc.molecule)

    return calc


def load_calc(
    path: str | Path,
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
            return _finalize_calc(calc)
        if root.suffix.lower() == ".json":
            artifacts: dict[str, Path] = {}
            if root.name.endswith(".calc_info.json"):
                artifacts["calc_info_json"] = root
            elif root.name.endswith("_mad_output.json"):
                artifacts["mad_output_json"] = root
            elif root.name == "output.json":
                artifacts["output_json"] = root
            else:
                raise ValueError(f"Unrecognized MADNESS JSON file: {root}")

            input_json = root.parent / "input.json"
            if input_json.exists():
                artifacts["input_json"] = input_json

            calc = Calculation(code="madness", root=root.parent, artifacts=artifacts, data={}, meta={})
            from gecko.plugins.madness.parse import parse_run

            parse_run(calc)
            return _finalize_calc(calc)
        raise ValueError(f"Expected a directory, got a file: {root}")

    if madness_can_load(root):
        calc = load_madness(root)
        return _finalize_calc(calc)

    if dalton_can_load(root):
        calc = load_dalton(root)
        return _finalize_calc(calc)

    raise ValueError(
        "Could not detect calculation type (madness/dalton) from directory. "
        f"Path: {root}"
    )


def load_calcs(
    path: str | Path,
) -> list[Calculation]:
    root = Path(path).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Path does not exist: {root}")

    if root.is_file():
        return [load_calc(root)]

    if madness_can_load(root):
        return [load_calc(root)]

    if dalton_can_load(root):
        return [load_calc(root)]

    raise ValueError(
        "Could not detect calculation type (madness/dalton) from directory. "
        f"Path: {root}"
    )
