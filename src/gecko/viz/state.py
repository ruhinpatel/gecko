from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np


METRIC_CHOICES = [
    ("rel error |dv| / ref_scale", "rel_err"),
    ("signed dv_parallel / ref_scale", "par_rel_signed"),
    ("angle(v_bas,v_ref) [deg]", "ang_err"),
]

FIELD_CHOICES = [
    ("Reference", "ref"),
    ("Basis", "bas"),
    ("Delta (bas-ref)", "dv"),
]


def metric_style(mode: str) -> Dict[str, Any]:
    signed = mode in ("dv_par_scalar", "par_rel_signed")
    sequential = not signed
    return {"signed": signed, "sequential": sequential}


def auto_clim(values: np.ndarray, *, mode: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    style = metric_style(mode)

    if finite.size == 0:
        return (-1.0, 1.0) if style["signed"] else (0.0, 1.0)

    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if style["signed"]:
        bound = max(abs(vmin), abs(vmax))
        if bound == 0:
            bound = 1.0
        return (-bound, bound)

    vmax = float(np.max(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return (0.0, vmax)


def default_state() -> Dict[str, Any]:
    return {
        "status": "starting",
        "status_color": "orange",
        "last_error": "",
        "mol": "H2O",
        "omega": 0.0,
        "ref_basis": "mra-high",
        "bas_basis_a": "aug-cc-pVDZ",
        "bas_basis_b": "d-aug-cc-pCVDZ",
        "lebedev_order": 29,
        "field_mode": "bas",
        "scalar_mode": "par_rel_signed",
        "glyph_factor": 0.12,
        "molecule_scale": 0.7,
        "atom_radius": 0.25,
        "coord_system": "as_is",
        "view_layout": "compare_bases",
        "scale_by_metric": False,
        "rel_norm": "global_rms",
        "show_axes": True,
        "metrics": {},
        "metric_rows": [],
        "posneg_yscale": "linear",
        "metric_plot_src": "",
        "metric_plot_error": "",
        "metric_headers": [
            {"text": "Metric", "value": "name"},
            {"text": "A", "value": "a"},
            {"text": "B", "value": "b"},
        ],
    }
