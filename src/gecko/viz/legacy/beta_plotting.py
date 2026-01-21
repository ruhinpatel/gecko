from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pyvista as pv
from scipy.integrate import lebedev_rule


# -------------------------
# Data + math primitives
# -------------------------

def beta_proj(beta: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Project a 3x3x3 beta tensor onto a direction.

    Returns a 3-vector:
        b_eff = sum_{j,k} beta_{i,j,k} E_j E_k

    Parameters
    ----------
    beta:
        Array shape (3,3,3).
    direction:
        Array shape (3,).
    """
    direction = np.asarray(direction, dtype=float)
    return np.tensordot(beta, np.outer(direction, direction), axes=([1, 2], [0, 1])).T


def beta_df_to_np(beta_df: pd.Series | Mapping[str, float]) -> np.ndarray:
    """Convert a 27-component beta Series (indexed by ijk like 'XYZ') to (3,3,3)."""
    ijk_map = {"X": 0, "Y": 1, "Z": 2}
    beta_np = np.zeros((3, 3, 3), dtype=float)
    for ijk, value in beta_df.items():
        i, j, k = ijk_map[ijk[0]], ijk_map[ijk[1]], ijk_map[ijk[2]]
        if pd.isna(value):
            beta_np[i, j, k] = 0.0
        else:
            beta_np[i, j, k] = float(value)
    return beta_np


def tensor_from_pivot(
    shg_pivot: pd.DataFrame, mol: str, basis: str, omega: float | int
) -> np.ndarray:
    """Fetch (mol,basis,omega) row from pivot and convert to tensor."""
    series = shg_pivot.loc[(mol, basis, omega)]
    return beta_df_to_np(series)


@dataclass(frozen=True)
class SphereQuadrature:
    points: np.ndarray  # (N,3)
    weights: np.ndarray  # (N,)
    order: int
    method: str = "lebedev"


def lebedev_sphere(order: int) -> SphereQuadrature:
    """Lebedev quadrature points on the unit sphere.

    Notes
    -----
    `scipy.integrate.lebedev_rule` returns points as (3,N). This wraps it as (N,3).
    """
    points, weights = lebedev_rule(order)
    return SphereQuadrature(points=points.T, weights=np.asarray(weights, dtype=float), order=order)


def project_tensor_on_points(beta: np.ndarray, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Project beta on each point direction.

    Returns
    -------
    vectors: (N,3)
    norms:   (N,)
    """
    # Reuse the shared tensor->field mapping used for error metrics and plotting.
    from field_error import evaluate_field

    vectors = evaluate_field(beta, points)
    norms = np.linalg.norm(vectors, axis=1)
    return vectors, norms


def integrated_average_magnitude(norms: np.ndarray, weights: np.ndarray) -> float:
    """Compute ⟨|b_eff|⟩ over the sphere using quadrature weights.

    This matches the existing notebook convention: (weights · norms) / (4π).
    """
    return float(np.asarray(weights) @ np.asarray(norms)) / (4.0 * math.pi)


# -------------------------
# Metrics (error mappings)
# -------------------------


@dataclass(frozen=True)
class FieldContext:
    points: np.ndarray  # (N,3)
    ref_vectors: np.ndarray  # (N,3)
    ref_norms: np.ndarray  # (N,)
    ref_scale: float
    basis_vectors: np.ndarray  # (N,3)
    basis_norms: np.ndarray  # (N,)


def make_field_context(
    *,
    points: np.ndarray,
    ref_vectors: np.ndarray,
    ref_norms: np.ndarray,
    ref_scale: float,
    basis_vectors: np.ndarray,
    basis_norms: np.ndarray,
) -> FieldContext:
    if ref_scale == 0:
        raise ValueError("ref_scale must be non-zero")
    return FieldContext(
        points=points,
        ref_vectors=ref_vectors,
        ref_norms=ref_norms,
        ref_scale=float(ref_scale),
        basis_vectors=basis_vectors,
        basis_norms=basis_norms,
    )


def metric_norm_error(ctx: FieldContext) -> np.ndarray:
    """(||b|| - ||b_ref||) / ref_scale."""
    return (ctx.basis_norms - ctx.ref_norms) / ctx.ref_scale


def metric_distance_error(ctx: FieldContext) -> np.ndarray:
    """||b - b_ref|| / ref_scale."""
    delta = ctx.basis_vectors - ctx.ref_vectors
    return np.linalg.norm(delta, axis=1) / ctx.ref_scale


def metric_signed_parallel_error(ctx: FieldContext, *, eps: float = 1e-16) -> np.ndarray:
    """Signed fractional error along the ref direction:

    e = (b · b_ref) / ||b_ref||^2 - 1

    Matches the notebook's `e_signed`.
    """
    denom = np.maximum(ctx.ref_norms**2, eps)
    dot = np.einsum("ij,ij->i", ctx.basis_vectors, ctx.ref_vectors)
    return dot / denom - 1.0


def metric_signed_log_parallel_error(
    ctx: FieldContext, *, mine: float = 0.01, maxe: float = 1.0
) -> np.ndarray:
    """Signed log mapping of |e| with saturation.

    Produces values in [-log10(maxe/mine), +log10(maxe/mine)] approximately.
    """
    e = metric_signed_parallel_error(ctx)
    mags = np.clip(np.abs(e), a_min=mine, a_max=maxe)
    # Map mine -> 0, maxe -> log10(maxe/mine)
    log_error = -np.log10(mine) + np.log10(mags)
    return np.sign(e) * log_error


def metric_basis_magnitude(ctx: FieldContext) -> np.ndarray:
    """||b|| (un-normalized). Useful for the unit-sphere field view."""
    return ctx.basis_norms


def metric_basis_magnitude_normalized(ctx: FieldContext) -> np.ndarray:
    """||b|| / ref_scale."""
    return ctx.basis_norms / ctx.ref_scale


@dataclass(frozen=True)
class MetricSpec:
    name: str
    label: str
    compute: Callable[..., np.ndarray]
    cmap: str
    symmetric: bool = False


METRICS: Dict[str, MetricSpec] = {
    "norm_error": MetricSpec(
        name="norm_error",
        label="(||b|| - ||b_ref||) / ref_scale",
        compute=lambda ctx, **_: metric_norm_error(ctx),
        cmap="coolwarm",
        symmetric=True,
    ),
    "distance_error": MetricSpec(
        name="distance_error",
        label="||b - b_ref|| / ref_scale",
        compute=lambda ctx, **_: metric_distance_error(ctx),
        cmap="viridis",
        symmetric=False,
    ),
    "signed_parallel_error": MetricSpec(
        name="signed_parallel_error",
        label="(b · b_ref)/||b_ref||^2 - 1",
        compute=lambda ctx, **kwargs: metric_signed_parallel_error(ctx, **kwargs),
        cmap="coolwarm",
        symmetric=True,
    ),
    "signed_log_parallel_error": MetricSpec(
        name="signed_log_parallel_error",
        label="sign(e) * log10(|e| / mine) (clipped)",
        compute=lambda ctx, **kwargs: metric_signed_log_parallel_error(ctx, **kwargs),
        cmap="CET_D1A",
        symmetric=True,
    ),
    "basis_magnitude": MetricSpec(
        name="basis_magnitude",
        label="||b||",
        compute=lambda ctx, **_: metric_basis_magnitude(ctx),
        cmap="viridis",
        symmetric=False,
    ),
    "basis_magnitude_normalized": MetricSpec(
        name="basis_magnitude_normalized",
        label="||b|| / ref_scale",
        compute=lambda ctx, **_: metric_basis_magnitude_normalized(ctx),
        cmap="viridis",
        symmetric=False,
    ),
}


def default_clim(values: np.ndarray, *, symmetric: bool) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).any():
        return (0.0, 1.0)

    vmin = float(np.nanmin(values))
    vmax = float(np.nanmax(values))
    if not symmetric:
        return (vmin, vmax)
    bound = max(abs(vmin), abs(vmax))
    return (-bound, bound)


def opacity_from_abs(values: np.ndarray, *, power: float = 4.0) -> np.ndarray:
    """Map |values| to [0,1] with a power curve."""
    v = np.abs(np.asarray(values, dtype=float))
    vmax = float(np.nanmax(v)) if np.size(v) else 1.0
    if vmax == 0:
        return np.zeros_like(v)
    return (v / vmax) ** power


# -------------------------
# Plot layers (PyVista)
# -------------------------

def create_molecule_plot(
    plotter: pv.Plotter,
    symbols: list[str],
    coords: np.ndarray,
    *,
    center: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    element_colors: Optional[Mapping[str, str]] = None,
    atom_radius: float = 0.2,
) -> pv.Plotter:
    """Draw a ball-model molecule into an existing plotter."""
    if element_colors is None:
        element_colors = {
            "H": "white",
            "C": "black",
            "N": "blue",
            "O": "red",
            "S": "yellow",
            "P": "orange",
            "Cl": "green",
            "F": "green",
            "Br": "brown",
            "I": "purple",
        }

    unique_elements = sorted(set(symbols))
    positions = np.asarray(coords, dtype=float) + np.array(center, dtype=float).reshape(1, 3)

    for elem in unique_elements:
        idx = [i for i, s in enumerate(symbols) if s == elem]
        pts = positions[idx]
        pd_ = pv.PolyData(pts)
        glyph = pd_.glyph(scale=False, orient=False, geom=pv.Sphere(radius=atom_radius))
        color = element_colors.get(elem, "gray")
        plotter.add_mesh(glyph, color=color)

    return plotter


def make_points_layer(
    *,
    points: np.ndarray,
    vectors: np.ndarray,
    scale_values: np.ndarray,
    scalars: np.ndarray,
    opacity: Optional[np.ndarray] = None,
    vectors_name: str = "vectors",
    scale_name: str = "scale",
    scalars_name: str = "scalars",
    opacity_name: str = "opacity",
) -> pv.PolyData:
    pd_ = pv.PolyData(np.asarray(points, dtype=float))

    vectors_arr = np.ascontiguousarray(np.asarray(vectors, dtype=float))
    if vectors_arr.ndim != 2 or vectors_arr.shape[1] != 3:
        raise ValueError(
            f"vectors must have shape (N,3); got {vectors_arr.shape}"
        )

    pd_[vectors_name] = vectors_arr
    pd_[scale_name] = np.asarray(scale_values, dtype=float)
    pd_[scalars_name] = np.asarray(scalars, dtype=float)
    if opacity is not None:
        pd_[opacity_name] = np.asarray(opacity, dtype=float)

    # Help PyVista pick up orientation arrays reliably.
    pd_.set_active_vectors(vectors_name)
    return pd_


def add_glyph_layer(
    plotter: pv.Plotter,
    layer: pv.PolyData,
    *,
    orient: str = "vectors",
    scale: str = "scale",
    factor: float = 0.05,
    scalars: str = "scalars",
    cmap: str = "viridis",
    clim: Optional[Tuple[float, float]] = None,
    opacity: str | float | None = None,
    show_scalar_bar: bool = True,
    scalar_bar_args: Optional[Dict[str, Any]] = None,
) -> None:
    glyphs = layer.glyph(orient=orient, scale=scale, factor=factor)

    kwargs: Dict[str, Any] = {
        "scalars": scalars,
        "cmap": cmap,
        "show_scalar_bar": show_scalar_bar,
    }
    if clim is not None:
        kwargs["clim"] = list(clim)
    if opacity is not None:
        kwargs["opacity"] = opacity
    if scalar_bar_args is not None:
        kwargs["scalar_bar_args"] = scalar_bar_args

    plotter.add_mesh(glyphs, **kwargs)


def plot_unit_sphere_field(
    *,
    mol: str,
    basis: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    mols: Mapping[str, Any],
    quad_order: int = 59,
    normalize_order: int = 125,
    normalize: bool = True,
    normalize_basis: str = "mra-high",
    glyph_factor: float = 0.05,
    cmap: str = "viridis",
    background: str = "white",
    show_axes: bool = True,
    off_screen: bool = False,
    show: bool = True,
    screenshot: str | None = None,
    camera_position: Any = "iso",
) -> pv.Plotter:
    """Single-panel unit-sphere field visualization.

    Uses Lebedev points on the unit sphere and draws glyph arrows oriented by
    the projected SHG field. Arrow lengths are scaled by ||b_eff||.

    Parameters
    ----------
    normalize:
        If True, normalize vectors by the sphere-average magnitude of
        `normalize_basis` (computed with `normalize_order`).

        This is intended to make different bases directly comparable by using
        the same per-(molecule, omega) reference scale (typically MRA).
    """
    quad = lebedev_sphere(quad_order)
    beta = tensor_from_pivot(shg_pivot, mol, basis, omega)
    vectors, norms = project_tensor_on_points(beta, quad.points)

    scale = 1.0
    if normalize:
        quad_n = lebedev_sphere(normalize_order)
        # Normalize using the same (mol, omega), but a fixed reference basis (typically MRA)
        beta_ref = tensor_from_pivot(shg_pivot, mol, normalize_basis, omega)
        _vectors_ref, norms_ref = project_tensor_on_points(beta_ref, quad_n.points)
        scale = integrated_average_magnitude(norms_ref, quad_n.weights)

        if (not np.isfinite(scale)) or scale == 0:
            scale = 1.0

    plotter = pv.Plotter(window_size=(500, 500), off_screen=off_screen)

    layer = make_points_layer(
        points=quad.points,
        vectors=vectors / scale,
        scale_values=norms,
        scalars=norms,
    )
    add_glyph_layer(
        plotter,
        layer,
        factor=glyph_factor / scale,
        scalars="scalars",
        cmap=cmap,
        clim=default_clim(norms, symmetric=False),
        scalar_bar_args={"title_font_size": 12, "label_font_size": 12},
    )

    symbols = mols[mol].symbols
    coords = mols[mol].geometry
    create_molecule_plot(plotter, symbols, coords * 0.7, atom_radius=0.25)
    plotter.add_text(f"{mol} | {basis} | ω={omega}", font_size=10)

    plotter.set_background(background)
    plotter.camera_position = camera_position
    if show_axes:
        plotter.show_axes()

    if show:
        if screenshot is not None:
            plotter.show(cpos=camera_position, screenshot=screenshot, auto_close=False)
        else:
            plotter.show(cpos=camera_position)
    return plotter


def build_unit_sphere_comparison_plotter(
    *,
    mol: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    mols: Mapping[str, Any],
    left_basis: str = "mra-high",
    right_basis: str = "aug-cc-pVDZ",
    quad_order: int = 59,
    normalize_order: int = 125,
    glyph_factor: float = 0.05,
    cmap: str = "viridis",
    background: str = "white",
    off_screen: bool = False,
    camera_position: Any = "iso",
) -> pv.Plotter:
    """Build (but do not render) a linked-camera two-panel comparison plotter.

    Intended notebook workflow:
      1) Build with off_screen=False, call show(return_cpos=True) and adjust view.
      2) Rebuild with off_screen=True and the captured cpos, then screenshot to PNG.
    """
    ref_scale = mra_reference_scale(
        mol=mol,
        omega=omega,
        shg_pivot=shg_pivot,
        reference_basis=left_basis,
        order=normalize_order,
    )
    if ref_scale == 0 or (not np.isfinite(ref_scale)):
        ref_scale = 1.0

    plotter = pv.Plotter(shape=(1, 2), window_size=(1000, 500), off_screen=off_screen)

    add_unit_sphere_field_to_subplot(
        plotter,
        row=0,
        col=0,
        mol=mol,
        basis=left_basis,
        omega=omega,
        shg_pivot=shg_pivot,
        mols=mols,
        quad_order=quad_order,
        ref_scale=ref_scale,
        glyph_factor=glyph_factor,
        cmap=cmap,
        show_scalar_bar=True,
    )

    add_unit_sphere_field_to_subplot(
        plotter,
        row=0,
        col=1,
        mol=mol,
        basis=right_basis,
        omega=omega,
        shg_pivot=shg_pivot,
        mols=mols,
        quad_order=quad_order,
        ref_scale=ref_scale,
        glyph_factor=glyph_factor,
        cmap=cmap,
        show_scalar_bar=True,
    )

    plotter.link_views()
    plotter.set_background(background)
    plotter.camera_position = camera_position
    plotter.add_text(f"{mol} | ω={omega} | scale={left_basis}", font_size=10)
    return plotter


def add_unit_sphere_field_to_subplot(
    plotter: pv.Plotter,
    *,
    row: int,
    col: int,
    mol: str,
    basis: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    mols: Mapping[str, Any],
    quad_order: int = 59,
    ref_scale: float = 1.0,
    glyph_factor: float = 0.05,
    cmap: str = "viridis",
    show_scalar_bar: bool = True,
) -> None:
    """Add a unit-sphere field plot into an existing subplot.

    Designed for multi-panel, linked-camera figures.
    """
    if ref_scale == 0 or (not np.isfinite(ref_scale)):
        ref_scale = 1.0

    plotter.subplot(row, col)

    quad = lebedev_sphere(quad_order)
    beta = tensor_from_pivot(shg_pivot, mol, basis, omega)
    vectors, norms = project_tensor_on_points(beta, quad.points)

    layer = make_points_layer(
        points=quad.points,
        vectors=vectors / ref_scale,
        scale_values=norms,
        scalars=norms,
    )

    add_glyph_layer(
        plotter,
        layer,
        factor=glyph_factor / ref_scale,
        scalars="scalars",
        cmap=cmap,
        clim=default_clim(norms, symmetric=False),
        show_scalar_bar=show_scalar_bar,
        scalar_bar_args={"title_font_size": 12, "label_font_size": 12},
    )

    symbols = mols[mol].symbols
    coords = mols[mol].geometry
    create_molecule_plot(plotter, symbols, coords * 0.7, atom_radius=0.25)
    plotter.add_text(f"{basis}", font_size=10)


def mra_reference_scale(
    *,
    mol: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    reference_basis: str = "mra-high",
    order: int = 125,
) -> float:
    """Compute the sphere-average magnitude scale for the MRA reference."""
    quad = lebedev_sphere(order)
    beta_ref = tensor_from_pivot(shg_pivot, mol, reference_basis, omega)
    _vec, norms = project_tensor_on_points(beta_ref, quad.points)
    return integrated_average_magnitude(norms, quad.weights)


def plot_unit_sphere_comparison(
    *,
    mol: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    mols: Mapping[str, Any],
    left_basis: str = "mra-high",
    right_basis: str = "aug-cc-pVDZ",
    quad_order: int = 59,
    normalize_order: int = 125,
    glyph_factor: float = 0.05,
    cmap: str = "viridis",
    background: str = "white",
    off_screen: bool = False,
    screenshot: str | None = None,
    camera_position: Any = "iso",
) -> pv.Plotter:
    """Two-panel unit-sphere comparison with linked camera.

    Both panels use the same normalization scale: the sphere-average magnitude
    of `left_basis` (default: MRA) for the same (mol, omega).

    If `off_screen=True` and `screenshot` is provided, renders directly to file.
    """
    plotter = build_unit_sphere_comparison_plotter(
        mol=mol,
        omega=omega,
        shg_pivot=shg_pivot,
        mols=mols,
        left_basis=left_basis,
        right_basis=right_basis,
        quad_order=quad_order,
        normalize_order=normalize_order,
        glyph_factor=glyph_factor,
        cmap=cmap,
        background=background,
        off_screen=off_screen,
        camera_position=camera_position,
    )

    if screenshot is not None:
        plotter.show(cpos=camera_position, screenshot=screenshot, auto_close=False)
    else:
        plotter.show(cpos=camera_position)

    return plotter


# -------------------------
# High-level composition
# -------------------------

def build_basis_field(
    *,
    shg_pivot: pd.DataFrame,
    mol: str,
    basis: str,
    omega: float | int,
    quad: SphereQuadrature,
) -> Tuple[np.ndarray, np.ndarray]:
    beta = tensor_from_pivot(shg_pivot, mol, basis, omega)
    return project_tensor_on_points(beta, quad.points)


def plot_basis_comparison_grid(
    *,
    mol: str,
    omega: float | int,
    shg_pivot: pd.DataFrame,
    mols: Mapping[str, Any],
    reference_basis: str = "mra-high",
    other_bases: list[str] = None,
    quad_order: int = 59,
    ref_scale_order: int = 125,
    glyph_factor: float = 0.05,
    metric: str = "signed_log_parallel_error",
    metric_kwargs: Optional[Dict[str, Any]] = None,
    metric_cmap: Optional[str] = None,
    max_cols: int = 3,
    background: str = "white",
) -> pv.Plotter:
    """Composable basis comparison grid.

    - Reference panel: shows the reference field colored by ||b||.
    - Other panels: show the basis field colored by the chosen error metric.

    Choose error visualization by passing `metric` from METRICS.
    """
    if other_bases is None:
        other_bases = ["aug-cc-pVDZ", "aug-cc-pVTZ"]
    if metric_kwargs is None:
        metric_kwargs = {}

    if metric not in METRICS:
        raise KeyError(f"Unknown metric '{metric}'. Available: {sorted(METRICS)}")

    metric_spec = METRICS[metric]
    cmap = metric_cmap or metric_spec.cmap

    quad = lebedev_sphere(quad_order)
    ref_vectors, ref_norms = build_basis_field(
        shg_pivot=shg_pivot, mol=mol, basis=reference_basis, omega=omega, quad=quad
    )

    # Reference normalization scale (separate order, matching the original notebook code)
    quad_scale = lebedev_sphere(ref_scale_order)
    ref_vectors_scale, ref_norms_scale = build_basis_field(
        shg_pivot=shg_pivot, mol=mol, basis=reference_basis, omega=omega, quad=quad_scale
    )
    ref_scale = integrated_average_magnitude(ref_norms_scale, quad_scale.weights)

    num_plots = 1 + len(other_bases)
    n_cols = min(num_plots, max_cols)
    n_rows = math.ceil(num_plots / n_cols)

    plotter = pv.Plotter(shape=(n_rows, n_cols), window_size=(350 * n_cols, 400 * n_rows))

    symbols = mols[mol].symbols
    coords = mols[mol].geometry

    # Reference panel
    plotter.subplot(0, 0)
    ref_layer = make_points_layer(
        points=quad.points,
        vectors=ref_vectors / ref_scale,
        scale_values=ref_norms,
        scalars=ref_norms,
    )
    add_glyph_layer(
        plotter,
        ref_layer,
        factor=glyph_factor / ref_scale,
        scalars="scalars",
        cmap="viridis",
        clim=default_clim(ref_norms, symmetric=False),
        scalar_bar_args={"title_font_size": 12, "label_font_size": 12},
    )
    create_molecule_plot(plotter, symbols, coords * 0.7, atom_radius=0.25)
    plotter.add_text(f"{reference_basis} (Reference)", font_size=10)

    # Error panels
    for idx, basis in enumerate(other_bases, start=1):
        row, col = divmod(idx, n_cols)
        plotter.subplot(row, col)

        basis_vectors, basis_norms = build_basis_field(
            shg_pivot=shg_pivot, mol=mol, basis=basis, omega=omega, quad=quad
        )

        ctx = make_field_context(
            points=quad.points,
            ref_vectors=ref_vectors,
            ref_norms=ref_norms,
            ref_scale=ref_scale,
            basis_vectors=basis_vectors,
            basis_norms=basis_norms,
        )

        values = metric_spec.compute(ctx, **metric_kwargs)
        clim = default_clim(values, symmetric=metric_spec.symmetric)
        opacity = opacity_from_abs(values, power=4.0)

        layer = make_points_layer(
            points=quad.points,
            vectors=basis_vectors / ref_scale,
            scale_values=basis_norms,
            scalars=values,
            opacity=opacity,
            scalars_name="metric",
        )

        add_glyph_layer(
            plotter,
            layer,
            factor=glyph_factor / ref_scale,
            scalars="metric",
            cmap=cmap,
            clim=clim,
            opacity="opacity",
            scalar_bar_args={"title_font_size": 12, "label_font_size": 12},
        )

        create_molecule_plot(plotter, symbols, coords * 0.7, atom_radius=0.25)
        plotter.add_text(basis, font_size=10)

    plotter.link_views()
    plotter.set_background(background)
    plotter.camera_position = "iso"
    plotter.show_axes_all()
    plotter.show(cpos="iso")

    return plotter
