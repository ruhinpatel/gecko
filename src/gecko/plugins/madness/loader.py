from __future__ import annotations

import json
from pathlib import Path

from gecko.core.model import Calculation
from gecko.plugins.madness.detect import can_load
from gecko.plugins.madness.parse import parse_run


def load(path: Path) -> Calculation:
    root = path.expanduser().resolve()

    if not can_load(root):
        raise ValueError(f"Not a MADNESS run directory: {root}")

    artifacts = _discover_artifacts(root)
    calc = Calculation(code="madness", root=root, artifacts=artifacts, data={}, meta={})

    # Fill data/meta using parse_run
    parse_run(calc)
    return calc


def _discover_artifacts(root: Path) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}

    # MADQC "marker": prefix.calc_info.json (prefix is user-defined or default "mad")
    # We'll pick the first match, and later make it smarter if needed.
    matches = list(root.glob("*.calc_info.json"))
    if matches:
        artifacts["calc_info_json"] = matches[0]

    # If you still produce something like "n12_mad_output.json" or "*_mad_output.json"
    # keep this for compatibility.
    mad_out = next(iter(root.glob("*_mad_output.json")), None)
    if mad_out:
        artifacts["mad_output_json"] = mad_out

    # Future / common location
    resp_meta = root / "responses" / "metadata.json"
    if resp_meta.exists():
        artifacts["responses_metadata_json"] = resp_meta

    # If you have precomputed SHG data saved somewhere (common in older workflows)
    for name in ("shg_ijk.csv", "beta.csv", "hyperpolarizability.csv"):
        p = root / name
        if p.exists():
            artifacts["beta_csv"] = p
            break
    # Keep old json marker too
    mad_out = next(iter(root.glob("output.json")), None)
    if mad_out:
        artifacts["output_json"] = mad_out

    input_json = root / "input.json"
    if input_json.exists():
        artifacts["input_json"] = input_json

    return artifacts
