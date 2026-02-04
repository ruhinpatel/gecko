from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

import math
import numpy as np

try:
    from scipy.integrate import lebedev_rule as _scipy_lebedev_rule  # type: ignore
except Exception:
    _scipy_lebedev_rule = None

Array = np.ndarray


@dataclass(frozen=True)
class LebedevGrid:
    n_hat: Array  # (N,3)
    w: Array  # (N,)
    order: int
    weight_sum: float


def load_lebedev_grid(order: int, *, weight_tol: float = 1e-8) -> LebedevGrid:
    """Load a unit-sphere quadrature grid (Lebedev when available)."""
    order = int(order)
    if order <= 0:
        raise ValueError(f"order must be positive; got {order}")

    if _scipy_lebedev_rule is not None:
        points, weights = _scipy_lebedev_rule(order)  # points: (3,N)
    else:
        # Fallback: quasi-uniform Fibonacci sphere with equal weights.
        # This is not a true Lebedev rule, but provides a stable, SciPy-free grid for visualization.
        n = order
        i = np.arange(n, dtype=float)
        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        z = 1.0 - (2.0 * i + 1.0) / n
        r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
        theta = golden_angle * i
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points = np.vstack([x, y, z])  # (3,N)
        weights = np.full(n, 4.0 * math.pi / n, dtype=float)

    n_hat = np.asarray(points, dtype=float).T
    w = np.asarray(weights, dtype=float)

    if n_hat.ndim != 2 or n_hat.shape[1] != 3:
        raise ValueError(f"Expected n_hat shape (N,3); got {n_hat.shape}")
    if w.ndim != 1 or w.shape[0] != n_hat.shape[0]:
        raise ValueError(f"Expected w shape (N,) matching n_hat; got {w.shape}")

    norms = np.linalg.norm(n_hat, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("n_hat contains non-finite values")
    if np.any(norms == 0):
        raise ValueError("n_hat contains zero-length directions")
    n_hat = n_hat / norms[:, None]

    weight_sum = float(np.sum(w))
    if abs(weight_sum - 4.0 * math.pi) > float(weight_tol):
        raise ValueError(
            f"Lebedev weight sum check failed: sum(w)={weight_sum} vs 4π={4.0 * math.pi} (tol={weight_tol})"
        )

    return LebedevGrid(n_hat=n_hat, w=w, order=int(order), weight_sum=weight_sum)


def evaluate_field(
    tensor: Array,
    n_hat: Array,
    *,
    mapping: str = "beta_proj",
    mapping_fn: Callable[[Array, Array], Array] | None = None,
    **mapping_kwargs: Any,
) -> Array:
    """Evaluate a vector field on the unit sphere."""
    n_hat = np.asarray(n_hat, dtype=float)
    if n_hat.ndim != 2 or n_hat.shape[1] != 3:
        raise ValueError(f"n_hat must have shape (N,3); got {n_hat.shape}")

    if mapping_fn is not None:
        v = mapping_fn(tensor, n_hat, **mapping_kwargs)
    else:
        if mapping != "beta_proj":
            raise KeyError(f"Unknown mapping '{mapping}'. Use mapping_fn for custom mapping.")
        beta = np.asarray(tensor, dtype=float)
        if beta.shape != (3, 3, 3):
            raise ValueError(f"beta tensor must have shape (3,3,3); got {beta.shape}")
        v = np.einsum("ijk,nj,nk->ni", beta, n_hat, n_hat)

    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"evaluate_field must return shape (N,3); got {v.shape}")
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class ErrorSettings:
    eps_rel: float = 1e-12
    eps_abs: float = 1e-16
    rel_norm: str = "global_rms"
    mask_threshold_fraction: float = 0.01
    enable_mask: bool = True
    component_metrics: bool = True
    radial_tangential_metrics: bool = True


def _percentile_finite(x: Array, q: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    return float(np.percentile(x, q))


def compute_error_fields(
    v_ref: Array,
    v_bas: Array,
    n_hat: Array,
    w: Array,
    *,
    settings: ErrorSettings | None = None,
) -> Tuple[Dict[str, Array], Dict[str, Any]]:
    if settings is None:
        settings = ErrorSettings()

    v_ref = np.asarray(v_ref, dtype=float)
    v_bas = np.asarray(v_bas, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)
    w = np.asarray(w, dtype=float)

    if v_ref.shape != v_bas.shape:
        raise ValueError(f"v_ref and v_bas must have same shape; got {v_ref.shape} vs {v_bas.shape}")
    if v_ref.ndim != 2 or v_ref.shape[1] != 3:
        raise ValueError(f"v_ref must have shape (N,3); got {v_ref.shape}")
    if n_hat.shape != v_ref.shape:
        raise ValueError(f"n_hat must have shape (N,3) matching fields; got {n_hat.shape}")
    if w.ndim != 1 or w.shape[0] != v_ref.shape[0]:
        raise ValueError(f"w must have shape (N,) matching fields; got {w.shape}")

    n_norm = np.linalg.norm(n_hat, axis=1)
    n_hat = n_hat / np.maximum(n_norm, 1e-300)[:, None]

    dv = v_bas - v_ref
    abs_err = np.linalg.norm(dv, axis=1)
    ref_mag = np.linalg.norm(v_ref, axis=1)
    bas_mag = np.linalg.norm(v_bas, axis=1)

    max_ref = float(np.nanmax(ref_mag)) if np.isfinite(ref_mag).any() else 0.0
    eps = max(float(settings.eps_abs), float(settings.eps_rel) * max_ref)

    def integrate(f: Array) -> float:
        f = np.asarray(f, dtype=float)
        return float(np.sum(w * f))

    weight_sum = float(np.sum(w))

    ref_mag_max = float(np.nanmax(ref_mag)) if np.isfinite(ref_mag).any() else 0.0
    ref_mag_mean = integrate(ref_mag) / (4.0 * math.pi)
    ref_mag_rms = math.sqrt(max(integrate(ref_mag**2) / (4.0 * math.pi), 0.0))

    rel_norm_kind = str(settings.rel_norm)
    if rel_norm_kind == "pointwise":
        denom = ref_mag + eps
    elif rel_norm_kind == "global_max":
        denom = ref_mag_max + eps
    elif rel_norm_kind == "global_mean":
        denom = ref_mag_mean + eps
    else:
        denom = ref_mag_rms + eps

    rel_err = abs_err / denom

    # parallel component
    n_hat_unit = n_hat
    v_ref_par = np.sum(v_ref * n_hat_unit, axis=1)
    v_bas_par = np.sum(v_bas * n_hat_unit, axis=1)
    dv_par = v_bas_par - v_ref_par
    dv_par_abs = np.abs(dv_par)
    par_rel = dv_par / denom

    # angle error
    dot = np.sum(v_ref * v_bas, axis=1)
    ref_norm = np.maximum(np.linalg.norm(v_ref, axis=1), 1e-300)
    bas_norm = np.maximum(np.linalg.norm(v_bas, axis=1), 1e-300)
    cosang = dot / (ref_norm * bas_norm)
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_err = np.degrees(np.arccos(cosang))

    arrays: Dict[str, Array] = {
        "ref_mag": ref_mag,
        "bas_mag": bas_mag,
        "abs_err": abs_err,
        "rel_err": rel_err,
        "dv_par_scalar": dv_par,
        "par_rel_signed": par_rel,
        "par_abs": dv_par_abs,
        "ang_err": ang_err,
        "n_hat": n_hat_unit,
    }

    metrics: Dict[str, Any] = {
        "settings": asdict(settings),
        "weight_sum": weight_sum,
        "ref_mag_max": ref_mag_max,
        "ref_mag_mean": ref_mag_mean,
        "ref_mag_rms": ref_mag_rms,
    }

    if settings.enable_mask:
        mask_threshold = float(settings.mask_threshold_fraction) * ref_mag_max
        mask = ref_mag >= mask_threshold
        arrays["mask"] = mask.astype(float)
        metrics["mask_threshold"] = mask_threshold
    else:
        arrays["mask"] = np.ones_like(ref_mag)
        metrics["mask_threshold"] = None

    if settings.component_metrics:
        arrays["ref_x"] = v_ref[:, 0]
        arrays["ref_y"] = v_ref[:, 1]
        arrays["ref_z"] = v_ref[:, 2]
        arrays["bas_x"] = v_bas[:, 0]
        arrays["bas_y"] = v_bas[:, 1]
        arrays["bas_z"] = v_bas[:, 2]
        arrays["dv_x"] = dv[:, 0]
        arrays["dv_y"] = dv[:, 1]
        arrays["dv_z"] = dv[:, 2]

    if settings.radial_tangential_metrics:
        # tangent component magnitude (removing radial component)
        v_ref_rad = v_ref_par[:, None] * n_hat_unit
        v_bas_rad = v_bas_par[:, None] * n_hat_unit
        v_ref_tan = v_ref - v_ref_rad
        v_bas_tan = v_bas - v_bas_rad
        arrays["ref_tan"] = np.linalg.norm(v_ref_tan, axis=1)
        arrays["bas_tan"] = np.linalg.norm(v_bas_tan, axis=1)
        arrays["dv_tan"] = np.linalg.norm(v_bas_tan - v_ref_tan, axis=1)
        arrays["ref_rad"] = np.abs(v_ref_par)
        arrays["bas_rad"] = np.abs(v_bas_par)
        arrays["dv_rad"] = np.abs(v_bas_par - v_ref_par)

    metrics["clim_percentiles"] = {
        "ref_mag": _percentile_finite(ref_mag, 99.0),
        "bas_mag": _percentile_finite(bas_mag, 99.0),
        "abs_err": _percentile_finite(abs_err, 99.0),
        "rel_err": _percentile_finite(rel_err, 99.0),
        "par_abs": _percentile_finite(dv_par_abs, 99.0),
        "ang_err": _percentile_finite(ang_err, 99.0),
    }

    return arrays, metrics


def _beta_df_to_np(beta_df) -> np.ndarray:
    ijk_map = {"X": 0, "Y": 1, "Z": 2}
    beta_np = np.zeros((3, 3, 3), dtype=float)
    for ijk, value in beta_df.items():
        i, j, k = ijk_map[ijk[0]], ijk_map[ijk[1]], ijk_map[ijk[2]]
        try:
            beta_np[i, j, k] = float(value)
        except Exception:
            beta_np[i, j, k] = 0.0
    return beta_np


def tensor_from_long(df, mol: str, basis: str, omega: float | int) -> np.ndarray:
    sub = df[
        (df["molecule"] == mol)
        & (df["basis"] == basis)
        & (df["omega"] == float(omega))
    ]
    if sub.shape[0] == 0:
        return np.zeros((3, 3, 3), dtype=float)
    series = {str(ijk): beta for ijk, beta in zip(sub["ijk"], sub["Beta"], strict=False)}
    return _beta_df_to_np(series)


__all__ = [
    "ErrorSettings",
    "compute_error_fields",
    "evaluate_field",
    "load_lebedev_grid",
    "tensor_from_long",
]
