from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

import json
import math

import numpy as np
from scipy.integrate import lebedev_rule


Array = np.ndarray


@dataclass(frozen=True)
class LebedevGrid:
    n_hat: Array  # (N,3)
    w: Array  # (N,)
    order: int
    weight_sum: float


def load_lebedev_grid(order: int, *, weight_tol: float = 1e-8) -> LebedevGrid:
    """Load a Lebedev grid (unit directions + weights).

    Parameters
    ----------
    order:
        Lebedev rule order passed to `scipy.integrate.lebedev_rule`.
    weight_tol:
        Absolute tolerance for validating `sum(w) ≈ 4π`.

    Returns
    -------
    LebedevGrid
        Contains deterministic-order `n_hat` with unit norm and weights `w`.
    """
    points, weights = lebedev_rule(int(order))  # points: (3,N)
    n_hat = np.asarray(points, dtype=float).T
    w = np.asarray(weights, dtype=float)

    if n_hat.ndim != 2 or n_hat.shape[1] != 3:
        raise ValueError(f"Expected n_hat shape (N,3); got {n_hat.shape}")
    if w.ndim != 1 or w.shape[0] != n_hat.shape[0]:
        raise ValueError(f"Expected w shape (N,) matching n_hat; got {w.shape}")

    # Ensure unit-norm directions.
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
    """Evaluate a vector field on the unit sphere.

    Default mapping is the tensor contraction used throughout this repo for SHG:

        v_i(n) = Σ_{j,k} β_{i,j,k} n_j n_k

    Parameters
    ----------
    tensor:
        Typically a (3,3,3) beta tensor.
    n_hat:
        Unit directions, shape (N,3).
    mapping:
        Name of built-in mapping. Currently supports: "beta_proj".
    mapping_fn:
        Optional callable override: mapping_fn(tensor, n_hat, **mapping_kwargs) -> (N,3).

    Returns
    -------
    v: (N,3)
    """
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
        # v[n,i] = sum_{j,k} beta[i,j,k] * n[n,j] * n[n,k]
        v = np.einsum("ijk,nj,nk->ni", beta, n_hat, n_hat)

    v = np.asarray(v, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"evaluate_field must return shape (N,3); got {v.shape}")
    return np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass(frozen=True)
class ErrorSettings:
    eps_rel: float = 1e-12
    eps_abs: float = 1e-16
    # How to normalize “relative error” quantities.
    #
    # - "pointwise": |dv(n)| / (|v_ref(n)| + eps)  [old behavior]
    # - "global_max": |dv(n)| / (max|v_ref| + eps)
    # - "global_mean": |dv(n)| / (mean|v_ref| + eps)  [Lebedev-weighted]
    # - "global_rms": |dv(n)| / (rms|v_ref| + eps)    [Lebedev-weighted]
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
    """Compute pointwise error arrays and Lebedev-weighted global metrics.

    Returns
    -------
    arrays: dict
        Numpy arrays suitable for saving to NPZ.
    metrics: dict
        JSON-serializable scalars and settings.
    """
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

    # Ensure n_hat is unit-norm (best-effort normalize).
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

    # Global reference magnitudes for more stable "relative" visualizations.
    # (Lebedev weights integrate over solid angle; normalize by 4π.)
    ref_mag_max = float(np.nanmax(ref_mag)) if np.isfinite(ref_mag).any() else 0.0
    ref_mag_mean = integrate(ref_mag) / (4.0 * math.pi)
    ref_mag_rms = math.sqrt(max(integrate(ref_mag**2) / (4.0 * math.pi), 0.0))

    rel_norm_kind = str(settings.rel_norm)
    if rel_norm_kind == "pointwise":
        denom_rel = ref_mag + eps
        rel_scale = float("nan")
    elif rel_norm_kind == "global_max":
        rel_scale = float(ref_mag_max)
        denom_rel = np.full_like(ref_mag, rel_scale + eps)
    elif rel_norm_kind == "global_mean":
        rel_scale = float(ref_mag_mean)
        denom_rel = np.full_like(ref_mag, rel_scale + eps)
    elif rel_norm_kind == "global_rms":
        rel_scale = float(ref_mag_rms)
        denom_rel = np.full_like(ref_mag, rel_scale + eps)
    else:
        raise ValueError(
            f"Unknown rel_norm='{settings.rel_norm}'. Use one of: "
            "pointwise, global_max, global_mean, global_rms"
        )

    # Main relative errors (for visualization / comparison)
    rel_err = abs_err / denom_rel

    # Angle error. Use a denominator clamp (not +eps) so v_bas==v_ref yields cos=1
    # wherever ||v_ref|| and ||v_bas|| are not near-zero.
    denom_ang = bas_mag * ref_mag
    denom_ang = np.maximum(denom_ang, eps)
    cosang = np.einsum("ij,ij->i", v_bas, v_ref) / denom_ang
    cosang = np.clip(cosang, -1.0, 1.0)
    ang_err = np.degrees(np.arccos(cosang))

    u_hat = v_ref / (ref_mag + eps)[:, None]
    dv_par_scalar = np.einsum("ij,ij->i", dv, u_hat)
    dv_par = dv_par_scalar[:, None] * u_hat
    dv_perp = dv - dv_par

    par_mag = np.abs(dv_par_scalar)
    perp_mag = np.linalg.norm(dv_perp, axis=1)

    par_rel = par_mag / denom_rel
    perp_rel = perp_mag / denom_rel
    par_rel_signed = dv_par_scalar / denom_rel

    L2_abs = math.sqrt(max(integrate(abs_err**2), 0.0))
    L2_ref = math.sqrt(max(integrate(ref_mag**2), 0.0))
    rel_L2 = L2_abs / (L2_ref + eps)

    L2_par = math.sqrt(max(integrate(dv_par_scalar**2), 0.0))
    L2_perp = math.sqrt(max(integrate(perp_mag**2), 0.0))
    ratio_perp_par = L2_perp / (L2_par + eps)

    metrics: Dict[str, Any] = {
        "weight_sum": weight_sum,
        "eps": eps,
        "rel_norm": rel_norm_kind,
        "rel_scale": rel_scale,
        "ref_mag_max": ref_mag_max,
        "ref_mag_mean": ref_mag_mean,
        "ref_mag_rms": ref_mag_rms,
        "L2_abs": L2_abs,
        "L2_ref": L2_ref,
        "rel_L2": rel_L2,
        "L2_par": L2_par,
        "L2_perp": L2_perp,
        "ratio_perp_par": ratio_perp_par,
        "max_abs": float(np.nanmax(abs_err)) if np.isfinite(abs_err).any() else float("nan"),
        "p95_abs": _percentile_finite(abs_err, 95),
        "max_rel": float(np.nanmax(rel_err)) if np.isfinite(rel_err).any() else float("nan"),
        "p95_rel": _percentile_finite(rel_err, 95),
        "mean_ang": integrate(ang_err) / (4.0 * math.pi),
        "p95_ang": _percentile_finite(ang_err, 95),
    }

    pos = dv_par_scalar > 0
    neg = dv_par_scalar < 0
    S_plus = float(np.sum(w[pos] * dv_par_scalar[pos])) if np.any(pos) else 0.0
    S_minus = float(np.sum(w[neg] * np.abs(dv_par_scalar[neg]))) if np.any(neg) else 0.0
    metrics.update(
        {
            "S_plus": S_plus,
            "S_minus": S_minus,
            "bias_ratio": (S_plus - S_minus) / (S_plus + S_minus + eps),
        }
    )

    arrays: Dict[str, Array] = {
        "n_hat": n_hat,
        "w": w,
        "v_ref": v_ref,
        "v_bas": v_bas,
        "dv": dv,
        "abs_err": abs_err,
        "ref_mag": ref_mag,
        "bas_mag": bas_mag,
        "rel_err": rel_err,
        "rel_err_pointwise": abs_err / (ref_mag + eps),
        "ang_err": ang_err,
        "dv_par_scalar": dv_par_scalar,
        "dv_par": dv_par,
        "dv_perp": dv_perp,
        "par_mag": par_mag,
        "perp_mag": perp_mag,
        "par_rel": par_rel,
        "perp_rel": perp_rel,
        "par_rel_signed": par_rel_signed,
        "par_rel_pointwise": par_mag / (ref_mag + eps),
        "perp_rel_pointwise": perp_mag / (ref_mag + eps),
        "par_rel_signed_pointwise": dv_par_scalar / (ref_mag + eps),
    }

    # Optional component-wise metrics
    if settings.component_metrics:
        for k, name in enumerate(("x", "y", "z")):
            L2_k = math.sqrt(max(integrate(dv[:, k] ** 2), 0.0))
            L2_ref_k = math.sqrt(max(integrate(v_ref[:, k] ** 2), 0.0))
            metrics[f"L2_{name}"] = L2_k
            metrics[f"rel_L2_{name}"] = L2_k / (L2_ref_k + eps)

    # Optional radial / tangential metrics
    if settings.radial_tangential_metrics:
        dv_r = np.einsum("ij,ij->i", dv, n_hat)
        dv_radial_vec = dv_r[:, None] * n_hat
        dv_tan = dv - dv_radial_vec

        arrays["dv_r"] = dv_r
        arrays["dv_tan"] = dv_tan

        metrics["L2_r"] = math.sqrt(max(integrate(dv_r**2), 0.0))
        metrics["L2_t"] = math.sqrt(max(integrate(np.linalg.norm(dv_tan, axis=1) ** 2), 0.0))

    # Masking near nodes
    if settings.enable_mask:
        positive = ref_mag[np.isfinite(ref_mag) & (ref_mag > 0)]
        if positive.size:
            thresh = float(settings.mask_threshold_fraction) * float(np.median(positive))
        else:
            thresh = float("inf")
        mask = ref_mag > thresh

        w_mask = w[mask]
        area_masked = float(np.sum(w_mask)) if w_mask.size else 0.0

        def integrate_masked(f: Array) -> float:
            if not np.any(mask):
                return 0.0
            f = np.asarray(f, dtype=float)
            return float(np.sum(w[mask] * f[mask]))

        L2_abs_m = math.sqrt(max(integrate_masked(abs_err**2), 0.0))
        L2_ref_m = math.sqrt(max(integrate_masked(ref_mag**2), 0.0))

        metrics.update(
            {
                "mask_threshold": thresh,
                "area_masked": area_masked,
                "masked_L2_abs": L2_abs_m,
                "masked_L2_ref": L2_ref_m,
                "masked_rel_L2": L2_abs_m / (L2_ref_m + eps),
                "masked_mean_rel": (integrate_masked(rel_err) / area_masked) if area_masked > 0 else float("nan"),
                "masked_p95_rel": _percentile_finite(rel_err[mask], 95) if np.any(mask) else float("nan"),
                "masked_max_rel": float(np.nanmax(rel_err[mask])) if np.any(mask) else float("nan"),
            }
        )
        arrays["mask"] = mask.astype(bool)

    metrics["settings"] = asdict(settings)
    return arrays, metrics


def save_npz(path: str | Path, arrays: Mapping[str, Array]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **dict(arrays))
    return path


def save_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _coerce(x: Any) -> Any:
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        return x

    coerced: Dict[str, Any] = {k: _coerce(v) for k, v in data.items()}
    with path.open("w", encoding="utf-8") as f:
        json.dump(coerced, f, indent=2, sort_keys=True)
        f.write("\n")
    return path


def write_outputs(
    out_dir: str | Path,
    *,
    arrays: Mapping[str, Array],
    metrics: Mapping[str, Any],
    identifiers: Mapping[str, Any] | None = None,
) -> Tuple[Path, Path]:
    """Write `errors.npz` and `metrics.json` under `out_dir`."""
    out_dir = Path(out_dir)
    errors_path = out_dir / "errors.npz"
    metrics_path = out_dir / "metrics.json"

    save_npz(errors_path, arrays)

    payload: Dict[str, Any] = dict(metrics)
    if identifiers is not None:
        payload["identifiers"] = dict(identifiers)

    save_json(metrics_path, payload)
    return errors_path, metrics_path


def plot_histogram(
    values: Array,
    *,
    path: str | Path,
    title: str,
    bins: int = 80,
    logy: bool = False,
) -> Path:
    """Optional helper: simple histogram plot using matplotlib."""
    import matplotlib.pyplot as plt

    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]

    fig, ax = plt.subplots(figsize=(5, 3.5), constrained_layout=True)
    ax.hist(v, bins=bins)
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    if logy:
        ax.set_yscale("log")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path
