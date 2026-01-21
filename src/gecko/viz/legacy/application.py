"""Trame server app for viewing SHG unit-sphere fields and error metrics.

This replaces the old demo that read a static `.vtu` file.

Run:
  python notebooks/scripts/application.py

Then open the printed URL in a browser.
"""

from __future__ import annotations

import argparse
import base64
import io
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

_DEBUG_STARTUP = os.environ.get("BETA_TRAME_DEBUG", "0") == "1"

_GLOBAL_CLIM_PERCENTILE = 99.0


def _dprint(msg: str) -> None:
    if _DEBUG_STARTUP:
        print(msg, flush=True)

try:
    import numpy as np
except ModuleNotFoundError as e:  # pragma: no cover
    raise SystemExit(
        "Missing Python dependencies (numpy). You're likely running with the wrong interpreter.\n"
        "Try one of:\n"
        "  - conda activate madqc && python notebooks/scripts/application.py\n"
        "  - conda run -n madqc python notebooks/scripts/application.py\n"
        "Or select the same environment you use in the notebooks."
    ) from e

_dprint("[startup] numpy imported")

from trame.app import get_server
from trame.ui.vuetify import SinglePageWithDrawerLayout
from trame.widgets import vtk, vuetify

_dprint("[startup] trame imported")

from trame_vtk.modules.vtk.serializers import configure_serializer
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource, vtkSphereSource
from vtkmodules.vtkRenderingAnnotation import (vtkAxesActor,
                                               vtkCornerAnnotation,
                                               vtkScalarBarActor)
from vtkmodules.vtkRenderingCore import (vtkActor, vtkCamera,
                                         vtkColorTransferFunction,
                                         vtkLogLookupTable, vtkPolyDataMapper,
                                         vtkRenderer, vtkRenderWindow,
                                         vtkRenderWindowInteractor,
                                         vtkTextActor)

_dprint("[startup] vtk modules imported")

# Rendering backend import (safe to include)
import vtkmodules.vtkRenderingOpenGL2  # noqa

configure_serializer(encode_lut=True, skip_light=True)

_dprint("[startup] serializer configured")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _ensure_import_paths() -> None:
    import sys

    repo_root = _repo_root()
    notebooks_root = repo_root / "notebooks"

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    if str(notebooks_root) not in sys.path:
        sys.path.insert(0, str(notebooks_root))


_ensure_import_paths()

from field_error import (ErrorSettings, compute_error_fields,  # noqa: E402
                         evaluate_field, load_lebedev_grid)

_SHG_CSV_PATH: Path | None = None
_MOLECULE_DIR: Path | None = None


def _default_shg_csv_path() -> Path:
    return _repo_root() / "data" / "csv_data" / "shg_ijk.csv"


def _resolve_shg_csv_path() -> Path:
    default_path = _default_shg_csv_path()
    if _SHG_CSV_PATH is not None:
        candidate = Path(_SHG_CSV_PATH).expanduser()
        if candidate.exists():
            return candidate
        if default_path.exists():
            print(
                f"Warning: SHG CSV not found at {candidate}; falling back to {default_path}",
                flush=True,
            )
            return default_path
        raise FileNotFoundError(
            f"SHG CSV not found at {candidate} and no default at {default_path}. "
            "Provide --shg_csv pointing to a valid file."
        )

    if default_path.exists():
        return default_path
    raise FileNotFoundError(
        f"Default SHG CSV not found at {default_path}. Provide --shg_csv to override."
    )


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1; use 0.0.0.0 for remote access)",
    )
    parser.add_argument(
        "--shg_csv",
        default="",
        help="Path to SHG CSV file to load (overrides built-in data)",
    )
    parser.add_argument(
        "--molecule_dir",
        default="",
        help="Path to directory containing .mol files (overrides built-in data)",
    )
    parser.add_argument(
        "--port",
        default=9010,
        type=int,
        help="Port to bind (default: 9010; use 0 to auto-pick a free port)",
    )
    return parser.parse_args()


def _select_free_port(host: str, preferred_port: int, *, max_tries: int = 50) -> int:
    import errno
    import socket

    bind_host = host or "127.0.0.1"
    if preferred_port == 0:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.bind((bind_host, 0))
            return int(sock.getsockname()[1])

    port = int(preferred_port)
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                sock.bind((bind_host, port))
                return port
            except OSError as exc:
                if exc.errno == errno.EADDRINUSE:
                    port += 1
                    continue
                raise

    raise RuntimeError(
        f"Could not find a free port starting at {preferred_port} after {max_tries} attempts"
    )


_CLI_ARGS = None
if __name__ == "__main__":
    _CLI_ARGS = _parse_args()
    if _CLI_ARGS.shg_csv:
        _SHG_CSV_PATH = Path(_CLI_ARGS.shg_csv).expanduser()
    if _CLI_ARGS.molecule_dir:
        _MOLECULE_DIR = Path(_CLI_ARGS.molecule_dir).expanduser()

@lru_cache(maxsize=1)
def _shg_long():
    import pandas as pd

    csv_path = _resolve_shg_csv_path()
    print(f"Loading SHG tensors from {csv_path} ...", flush=True)
    shg_ijk = pd.read_csv(csv_path)
    # Expected columns: molecule, basis, omega, ijk, Beta
    # Normalize string columns so basis-family parsing and lookups are robust.
    for col in ("molecule", "basis", "ijk"):
        if col in shg_ijk.columns:
            shg_ijk[col] = (
                shg_ijk[col]
                .astype(str)
                .str.strip()
                .str.replace(r"\s+", "", regex=True)
            )
    if "omega" in shg_ijk.columns:
        shg_ijk["omega"] = shg_ijk["omega"].astype(float)
    return shg_ijk


def _beta_df_to_np(beta_df) -> np.ndarray:
    ijk_map = {"x": 0, "y": 1, "z": 2}
    beta_np = np.zeros((3, 3, 3), dtype=float)
    for ijk, value in beta_df.items():
        i, j, k = ijk_map[ijk[0].lower()],ijk_map[ijk[1].lower()],ijk_map[ijk[2].lower()]
        try:
            beta_np[i, j, k] = float(value)
        except Exception:
            beta_np[i, j, k] = 0.0
    return beta_np


def _tensor_from_long(shg_long, mol: str, basis: str, omega: float | int) -> np.ndarray:
    sub = shg_long[
        (shg_long["molecule"] == mol)
        & (shg_long["basis"] == basis)
        & (shg_long["omega"] == float(omega))
    ]
    if sub.shape[0] == 0:
        raise KeyError(f"No SHG tensor found for (mol={mol}, basis={basis}, omega={omega})")
    series = {str(ijk): beta for ijk, beta in zip(sub["ijk"], sub["Beta"], strict=False)}
    return _beta_df_to_np(series)


@lru_cache(maxsize=1)
def _data():
    # Intentionally only load the SHG tensor table here.
    return _shg_long()


@lru_cache(maxsize=256)
def _load_molecule(mol_name: str):
    import qcelemental as _qcel

    from gecko.plugins.madness.legacy.madness_molecule import MADMolecule

    mol_dir = _MOLECULE_DIR or (_repo_root() / "data" / "molecules")
    mol_path = mol_dir / f"{mol_name}.mol"
    madmol = MADMolecule()
    madmol.from_molfile(mol_path)
    mol_json = madmol.to_json()
    return _qcel.models.Molecule(symbols=mol_json["symbols"], geometry=mol_json["geometry"])


# -----------------------------------------------------------------------------
# VTK scene
# -----------------------------------------------------------------------------


def _new_scene(
    *,
    renderWindow: vtkRenderWindow,
    viewport: Tuple[float, float, float, float],
    background: Tuple[float, float, float] = (0.08, 0.08, 0.10),
) -> Dict[str, Any]:
    """Create a renderer + actors within a shared renderWindow.

    We use two renderers in a single render window (left/right viewports) and
    share a single camera to guarantee linked camera interaction.
    """

    renderer = vtkRenderer()
    renderer.SetBackground(float(background[0]), float(background[1]), float(background[2]))
    renderer.SetViewport(float(viewport[0]), float(viewport[1]), float(viewport[2]), float(viewport[3]))
    renderWindow.AddRenderer(renderer)

    # Always render *something* even if glyphs/molecule fail.
    sphere_source = vtkSphereSource()
    sphere_source.SetRadius(1.0)
    sphere_source.SetThetaResolution(48)
    sphere_source.SetPhiResolution(48)

    sphere_mapper = vtkPolyDataMapper()
    sphere_mapper.SetInputConnection(sphere_source.GetOutputPort())


    sphere_actor = vtkActor()
    sphere_actor.SetMapper(sphere_mapper)
    sphere_actor.GetProperty().SetRepresentationToWireframe()
    sphere_actor.GetProperty().SetColor(0.75, 0.75, 0.80)
    sphere_actor.GetProperty().SetOpacity(0.10)
    renderer.AddActor(sphere_actor)

    # Coordinate axes (3D actor) so the frame is visible.
    axes_actor = vtkAxesActor()
    axes_actor.SetTotalLength(0.55, 0.55, 0.55)
    axes_actor.SetShaftTypeToCylinder()
    axes_actor.SetCylinderRadius(0.02)
    axes_actor.AxisLabelsOn()
    renderer.AddActor(axes_actor)

    # Scalar bar to show the active color range for the glyphs.
    scalar_bar = vtkScalarBarActor()
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLabelFormat("%-#6.3g")
    scalar_bar.SetPosition(0.83, 0.10)
    scalar_bar.SetPosition2(0.15, 0.80)
    scalar_bar.VisibilityOff()
    renderer.AddViewProp(scalar_bar)

    # Big, always-visible title in the upper-left of each viewport.
    title_actor = vtkTextActor()
    title_actor.SetInput("")
    title_actor.GetPositionCoordinate().SetCoordinateSystemToNormalizedViewport()
    title_actor.SetPosition(0.02, 0.98)
    title_actor.GetTextProperty().SetJustificationToLeft()
    title_actor.GetTextProperty().SetVerticalJustificationToTop()
    title_actor.GetTextProperty().SetFontSize(22)
    title_actor.GetTextProperty().BoldOn()
    title_actor.GetTextProperty().SetColor(0.92, 0.92, 0.92)
    renderer.AddViewProp(title_actor)

    # Show coordinate system / view context.
    corner = vtkCornerAnnotation()
    corner.SetLinearFontScaleFactor(2)
    corner.SetNonlinearFontScaleFactor(1)
    corner.SetMaximumFontSize(26)
    corner.GetTextProperty().SetColor(0.92, 0.92, 0.92)
    corner.SetText(2, "")
    renderer.AddViewProp(corner)

    return {
        "renderer": renderer,
        "renderWindow": renderWindow,
        "sphere_actor": sphere_actor,
        "axes_actor": axes_actor,
        "scalar_bar": scalar_bar,
        "title_actor": title_actor,
        "corner": corner,
        "glyph_actor": None,
        "mol_actors": [],
    }


def _create_render_window() -> vtkRenderWindow:
    """Create a render window compatible with headless environments.

    Even when using client-side VTK.js rendering, VTK may try to create an X-backed
    OpenGL context when instantiating a plain `vtkRenderWindow()` on Linux.
    Prefer EGL offscreen rendering when no DISPLAY is available.
    """

    egl_mode = os.environ.get("BETA_VTK_EGL", "auto").strip().lower()
    prefer_egl = egl_mode != "0"
    require_egl = egl_mode == "1"

    if prefer_egl:
        try:
            from vtkmodules.vtkRenderingOpenGL2 import vtkEGLRenderWindow

            rw = vtkEGLRenderWindow()
            rw.SetOffScreenRendering(1)
            _dprint("[startup] using vtkEGLRenderWindow (offscreen)")
            return rw
        except Exception as exc:
            if require_egl:
                raise RuntimeError(
                    "BETA_VTK_EGL=1 requested EGL offscreen rendering, but vtkEGLRenderWindow "
                    f"could not be created: {exc}"
                ) from exc
            _dprint(f"[startup] vtkEGLRenderWindow unavailable, falling back: {exc}")

    # Fall back to the platform-default render window (may require a working DISPLAY).
    rw = vtkRenderWindow()
    _dprint("[startup] using default vtkRenderWindow")
    return rw


_dprint("[startup] creating VTK render window")
renderWindow = _create_render_window()

# Keep an interactor attached for compatibility with some trame-vtk helpers.
renderWindowInteractor = vtkRenderWindowInteractor()
renderWindowInteractor.SetRenderWindow(renderWindow)

_shared_camera = vtkCamera()
_camera_initialized = False


def _rotate_beta(beta: np.ndarray, R: np.ndarray) -> np.ndarray:
        """Rotate beta tensor into a new coordinate system.

        Conventions:
        - Vectors are treated as column vectors in the math, but in this code we
            often store coordinates as row vectors.
        - If we use `coords_new = coords_old @ R` for row vectors, then the column
            transform is `v_new = R.T @ v_old`.
        - Therefore beta transforms as:
                beta_new[i,j,k] = Rt[i,a] Rt[j,b] Rt[k,c] beta_old[a,b,c]
            where Rt = R.T.
        """
        beta = np.asarray(beta, dtype=float)
        R = np.asarray(R, dtype=float)
        Rt = R.T
        return np.einsum("ia,jb,kc,abc->ijk", Rt, Rt, Rt, beta, optimize=True)


def _principal_axes_rotation(mol) -> np.ndarray:
        """Compute a deterministic principal-axes frame with a phase convention.

        - Uses an inertia-tensor principal axes definition.
        - Enforces a right-handed basis.
        - Fixes axis signs by aligning axes with a reference direction derived from
            the heaviest atom.
        """

        import numpy as _np
        import qcelemental as _qcel

        symbols = list(mol.symbols)
        coords = _np.asarray(mol.geometry, dtype=float)
        coords = coords - _np.mean(coords, axis=0, keepdims=True)

        masses = _np.array([
                float(_qcel.periodictable.to_mass(sym)) if sym is not None else 1.0 for sym in symbols
        ])
        if not _np.all(_np.isfinite(masses)):
                masses = _np.ones(len(symbols), dtype=float)

        # Inertia tensor about the origin: I = Σ m (r^2 I - r r^T)
        I = _np.zeros((3, 3), dtype=float)
        for r, m in zip(coords, masses, strict=False):
                r2 = float(_np.dot(r, r))
                I += m * (r2 * _np.eye(3) - _np.outer(r, r))

        evals, evecs = _np.linalg.eigh(I)
        order = _np.argsort(evals)
        R = evecs[:, order]

        # Phase convention: use the direction to the heaviest atom.
        ref_idx = int(_np.argmax(masses)) if masses.size else 0
        ref = coords[ref_idx] if coords.size else _np.array([1.0, 0.0, 0.0])
        if _np.linalg.norm(ref) < 1e-12:
                ref = _np.array([1.0, 0.0, 0.0])
        dots = R.T @ ref
        signs = _np.where(dots >= 0, 1.0, -1.0)
        R = R * signs

        # Ensure right-handed
        if _np.linalg.det(R) < 0:
                R[:, 2] *= -1.0
        return R

_scene_left = _new_scene(renderWindow=renderWindow, viewport=(0.0, 0.0, 0.5, 1.0))
_scene_mid = _new_scene(renderWindow=renderWindow, viewport=(0.5, 0.0, 1.0, 1.0))
_scene_right = _new_scene(renderWindow=renderWindow, viewport=(0.0, 0.0, 0.0, 0.0))
_scene_left["renderer"].SetActiveCamera(_shared_camera)
_scene_mid["renderer"].SetActiveCamera(_shared_camera)
_scene_right["renderer"].SetActiveCamera(_shared_camera)

_dprint("[startup] VTK render window created")


def _polydata_from_points(points: np.ndarray) -> vtkPolyData:
    points = np.asarray(points, dtype=float)
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points, deep=True))
    pd = vtkPolyData()
    pd.SetPoints(vtk_pts)
    return pd


def _set_vectors(pd: vtkPolyData, name: str, vectors: np.ndarray) -> None:
    vectors = np.asarray(vectors, dtype=float)
    vtk_arr = numpy_to_vtk(vectors, deep=True)
    vtk_arr.SetName(name)
    vtk_arr.SetNumberOfComponents(3)
    pd.GetPointData().SetVectors(vtk_arr)


def _set_scalars(pd: vtkPolyData, name: str, scalars: np.ndarray) -> None:
    scalars = np.asarray(scalars, dtype=float).reshape(-1)
    vtk_arr = numpy_to_vtk(scalars, deep=True)
    vtk_arr.SetName(name)
    vtk_arr.SetNumberOfComponents(1)
    pd.GetPointData().SetScalars(vtk_arr)


def _add_array(pd: vtkPolyData, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    vtk_arr = numpy_to_vtk(arr, deep=True)
    vtk_arr.SetName(name)
    pd.GetPointData().AddArray(vtk_arr)


def _element_rgb(symbol: str) -> Tuple[float, float, float]:
    table = {
        "H": (1.0, 1.0, 1.0),
        "C": (0.1, 0.1, 0.1),
        "N": (0.2, 0.2, 1.0),
        "O": (1.0, 0.1, 0.1),
        "S": (1.0, 0.9, 0.1),
        "P": (1.0, 0.5, 0.1),
        "Cl": (0.1, 0.8, 0.1),
        "F": (0.1, 0.8, 0.1),
        "Br": (0.6, 0.3, 0.1),
        "I": (0.5, 0.2, 0.8),
    }
    return table.get(symbol, (0.6, 0.6, 0.6))


def _default_clim(values: np.ndarray, *, symmetric: bool) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (0.0, 1.0)
    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if symmetric:
        bound = max(abs(vmin), abs(vmax))
        if bound == 0:
            bound = 1.0
        return (-bound, bound)
    if vmin == vmax:
        vmax = vmin + 1.0
    return (vmin, vmax)


def _auto_clim(values: np.ndarray, *, mode: str) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    style = _metric_style(mode)

    if finite.size == 0:
        return (-1.0, 1.0) if style["signed"] else (0.0, 1.0)

    vmin, vmax = float(np.min(finite)), float(np.max(finite))
    if style["signed"]:
        bound = max(abs(vmin), abs(vmax))
        if bound == 0:
            bound = 1.0
        return (-bound, bound)

    # Unsigned metrics: enforce 0 -> max
    vmax = float(np.max(finite))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    return (0.0, vmax)


def _metric_style(mode: str) -> Dict[str, Any]:
    """Return styling metadata for a scalar metric."""
    signed = mode in ("dv_par_scalar", "par_rel_signed")
    # Everything else currently displayed is non-negative.
    sequential = not signed
    return {"signed": signed, "sequential": sequential}


def _build_lut(*, kind: str, clim: Tuple[float, float], scalar_mode: str) -> vtkLookupTable:
    """Create a LUT consistent with our metric semantics."""

    vmin, vmax = float(clim[0]), float(clim[1])
    if vmin == vmax:
        vmax = vmin + 1.0

    # Base colors from a color transfer function (RGB).
    ctf = vtkColorTransferFunction()
    ctf.SetRange(vmin, vmax)
    if kind == "diverging":
        ctf.SetColorSpaceToDiverging()
        ctf.AddRGBPoint(vmin, 0.230, 0.299, 0.754)
        ctf.AddRGBPoint(0.5 * (vmin + vmax), 0.865, 0.865, 0.865)
        ctf.AddRGBPoint(vmax, 0.706, 0.016, 0.150)
    else:
        ctf.SetColorSpaceToRGB()
        ctf.AddRGBPoint(vmin, 0.267, 0.005, 0.329)
        ctf.AddRGBPoint(vmin + 0.25 * (vmax - vmin), 0.283, 0.141, 0.458)
        ctf.AddRGBPoint(vmin + 0.50 * (vmax - vmin), 0.254, 0.265, 0.530)
        ctf.AddRGBPoint(vmin + 0.75 * (vmax - vmin), 0.207, 0.372, 0.553)
        ctf.AddRGBPoint(vmax, 0.993, 0.906, 0.144)

    table_size = 256
    lut = vtkLookupTable()
    lut.SetNumberOfTableValues(table_size)
    lut.SetRange(vmin, vmax)
    lut.Build()

    for i in range(table_size):
        t = i / float(table_size - 1)
        x = vmin + t * (vmax - vmin)
        r, g, b = ctf.GetColor(x)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)

    return lut


def _build_glyph_actor(
    *,
    n_hat: np.ndarray,
    vectors: np.ndarray,
    scalars: np.ndarray,
    scale: np.ndarray,
    glyph_factor: float,
    clim: Tuple[float, float],
    lut_kind: str,
    scalar_mode: str,
) -> vtkActor:
    pd = _polydata_from_points(n_hat)
    _set_vectors(pd, "vectors", vectors)

    # Preserve the color scalars array even though we also need an active scalar
    # array ('scale') for glyph sizing.
    scalars = np.asarray(scalars, dtype=float).reshape(-1)
    _add_array(pd, "scalars", scalars)

    # Normalize scale so glyph sizing is stable.
    scale = np.asarray(scale, dtype=float).reshape(-1)
    scale_max = float(np.max(scale)) if scale.size else 0.0
    if not np.isfinite(scale_max) or scale_max <= 0:
        scale_norm = np.ones_like(scale)
    else:
        scale_norm = scale / scale_max
        # Preserve explicit masking via zero scale.
        scale_norm = np.where(scale_norm <= 0.0, 0.0, np.clip(scale_norm, 0.05, 1.0))

    # Use 'scale' as the active scalars for vtkGlyph3D scaling.
    _set_scalars(pd, "scale", scale_norm)

    arrow = vtkArrowSource()
    arrow.SetTipLength(0.35)
    arrow.SetTipRadius(0.10)
    arrow.SetShaftRadius(0.03)

    glyph = vtkGlyph3D()
    glyph.SetInputData(pd)
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.OrientOn()
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByScalar()
    glyph.SetScaleFactor(float(glyph_factor))

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.ScalarVisibilityOn()
    mapper.SetScalarModeToUsePointFieldData()
    mapper.SelectColorArray("scalars")

    vmin, vmax = float(clim[0]), float(clim[1])
    if vmin == vmax:
        vmax = vmin + 1.0
    lut = _build_lut(kind=str(lut_kind), clim=(vmin, vmax), scalar_mode=str(scalar_mode))
    mapper.SetLookupTable(lut)
    mapper.SetScalarRange(vmin, vmax)

    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def _build_atom_actor(points: np.ndarray, *, radius: float, rgb: Tuple[float, float, float]) -> vtkActor:
    pd = _polydata_from_points(points)
    sphere = vtkSphereSource()
    sphere.SetRadius(float(radius))
    sphere.SetThetaResolution(24)
    sphere.SetPhiResolution(24)

    glyph = vtkGlyph3D()
    glyph.SetInputData(pd)
    glyph.SetSourceConnection(sphere.GetOutputPort())
    glyph.ScalingOff()
    glyph.OrientOff()

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    actor = vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetColor(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    return actor


# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------


server = get_server(client_type="vue2")
state, ctrl = server.state, server.controller

# During module import, callbacks may run before the VTK view is created.
# Provide safe defaults until `ctrl.view_update` is set to the actual view's update.
ctrl.view_update = lambda **_: None


def _reset_camera() -> None:
    global _camera_initialized
    _scene_left["renderer"].ResetCamera()
    _camera_initialized = True
    renderWindow.Modified()
    ctrl.view_update()


ctrl.reset_camera = _reset_camera

state.setdefault("status", "starting")
state.setdefault("status_color", "orange")
state.setdefault("last_error", "")


def _basis_list() -> List[str]:
    df = _data()
    return sorted(set(str(x) for x in df["basis"].unique()))


def _molecule_list() -> List[str]:
    df = _data()
    return sorted(set(str(x) for x in df["molecule"].unique()))


def _omega_list(mol: str) -> List[float]:
    df = _data()
    sub = df[df["molecule"] == mol]
    return sorted(set(float(x) for x in sub["omega"].unique()))


METRIC_CHOICES = [
    ("|v_ref|", "ref_mag"),
    ("|v_bas|", "bas_mag"),
    ("|dv|", "abs_err"),
    ("|dv| / ref_scale", "rel_err"),
    ("angle(v_bas,v_ref) [deg]", "ang_err"),
    ("signed dv_parallel", "dv_par_scalar"),
    ("signed dv_parallel/ref_scale", "par_rel_signed"),
    ("|dv_perp|", "perp_mag"),
    ("|dv_parallel|/ref_scale", "par_rel"),
    ("|dv_perp|/ref_scale", "perp_rel"),
]

FIELD_CHOICES = [
    ("Reference", "ref"),
    ("Basis", "bas"),
    ("Delta (bas-ref)", "dv"),
]


def _scalar_mode_label(mode: str) -> str:
    for label, value in METRIC_CHOICES:
        if value == mode:
            return str(label)
    return str(mode)


def _rel_norm_label() -> str:
    kind = str(state.rel_norm)
    if kind == "pointwise":
        return "|v_ref(n)|"
    if kind == "global_max":
        return "max(|v_ref|)"
    if kind == "global_mean":
        return "mean(|v_ref|)"
    if kind == "global_rms":
        return "rms(|v_ref|)"
    return kind


def _finite_concat(chunks: List[np.ndarray]) -> np.ndarray:
    if not chunks:
        return np.asarray([], dtype=float)
    arr = np.concatenate([np.asarray(x, dtype=float).reshape(-1) for x in chunks], axis=0)
    return arr[np.isfinite(arr)]


@lru_cache(maxsize=64)
def _global_metric_clims(
    mol: str,
    omega: float,
    ref_basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
) -> Dict[str, Tuple[float, float]]:
    """Compute robust clims for each metric across all basis sets.

    Returned clims are intended for visualization only.
    """
    grid = load_lebedev_grid(int(lebedev_order))
    df = _data()

    beta_ref = _tensor_from_long(df, mol, ref_basis, float(omega))

    R = None
    if coord_system == "principal_axes":
        qmol = _load_molecule(mol)
        R = _principal_axes_rotation(qmol)
        beta_ref = _rotate_beta(beta_ref, R)

    if isinstance(grid, tuple) and len(grid) == 2:
        n_hat = np.asarray(grid[0], dtype=float)
        w = np.asarray(grid[1], dtype=float)
    else:
        n_hat = np.asarray(grid.n_hat, dtype=float)
        w = np.asarray(grid.w, dtype=float)

    v_ref = evaluate_field(beta_ref, n_hat)
    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(rel_norm),
    )

    # Collect values for each metric across all basis sets.
    metric_values: Dict[str, List[np.ndarray]] = {}
    for basis in _basis_list():
        if str(basis) == str(ref_basis):
            continue
        try:
            beta_bas = _tensor_from_long(df, mol, str(basis), float(omega))
        except Exception:
            continue
        if R is not None:
            beta_bas = _rotate_beta(beta_bas, R)
        v_bas = evaluate_field(beta_bas, n_hat)
        arrays, _ = compute_error_fields(v_ref, v_bas, n_hat, w, settings=settings)

        for key in (
            "ref_mag",
            "bas_mag",
            "abs_err",
            "rel_err",
            "ang_err",
            "dv_par_scalar",
            "par_rel_signed",
            "perp_mag",
            "par_rel",
            "perp_rel",
        ):
            if key in arrays:
                metric_values.setdefault(key, []).append(np.asarray(arrays[key]))

    clims: Dict[str, Tuple[float, float]] = {}
    for key, chunks in metric_values.items():
        vals = _finite_concat(chunks)
        style = _metric_style(key)
        if vals.size == 0:
            clims[key] = (-1.0, 1.0) if style["signed"] else (0.0, 1.0)
            continue
        if style["signed"]:
            bound = float(np.percentile(np.abs(vals), _GLOBAL_CLIM_PERCENTILE))
            if not np.isfinite(bound) or bound <= 0:
                bound = 1.0
            clims[key] = (-bound, bound)
        else:
            vmax = float(np.percentile(vals, _GLOBAL_CLIM_PERCENTILE))
            if not np.isfinite(vmax) or vmax <= 0:
                vmax = 1.0
            clims[key] = (0.0, vmax)

    # ref_mag is common; if missing, fall back to local behavior.
    return clims


# -----------------------------------------------------------------------------
# Metric plots across basis sets
# -----------------------------------------------------------------------------


_PLOT_FAMILY_ORDER: List[str] = [
    "aug-cc-pVnZ",
    "aug-cc-pCVnZ",
    "d-aug-cc-pVnZ",
    "d-aug-cc-pCVnZ",
]

_PLOT_CARD_ORDER: List[str] = ["DZ", "TZ", "QZ"]


def _basis_family_and_cardinal(basis: str) -> Tuple[str | None, str | None]:
    """Return (family, cardinal) for a basis name.

    Supported families:
      aug-cc-pVnZ, aug-cc-pCVnZ, d-aug-cc-pVnZ, d-aug-cc-pCVnZ
    Supported cardinals:
      DZ/TZ/QZ
    """
    b = re.sub(r"\s+", "", str(basis)).strip()

    # pV family: aug-cc-pVDZ / aug-cc-pVTZ / aug-cc-pVQZ (and d-aug-...)
    m = re.match(r"^(d-aug|aug)-cc-pV(D|T|Q)Z$", b)
    if m:
        aug_prefix = "d-aug" if m.group(1) == "d-aug" else "aug"
        card = f"{m.group(2)}Z"
        return f"{aug_prefix}-cc-pVnZ", card

    # pCV family: aug-cc-pCVDZ / aug-cc-pCVTZ / aug-cc-pCVQZ (and d-aug-...)
    m = re.match(r"^(d-aug|aug)-cc-pCV(D|T|Q)Z$", b)
    if m:
        aug_prefix = "d-aug" if m.group(1) == "d-aug" else "aug"
        card = f"{m.group(2)}Z"
        return f"{aug_prefix}-cc-pCVnZ", card

    return None, None


PLOT_METRIC_CHOICES: List[Tuple[str, str]] = [
    ("L2 relative error (rel_L2)", "rel_L2"),
    ("Mean signed rel parallel (mean_par_rel_signed)", "mean_par_rel_signed"),
    (
        "Mean + / Mean − signed rel parallel (posneg_par_rel_signed)",
        "posneg_par_rel_signed",
    ),
    ("P95 signed rel parallel (p95_par_rel_signed)", "p95_par_rel_signed"),
    ("Mean angle [deg] (mean_ang)", "mean_ang"),
    ("P95 angle [deg] (p95_ang)", "p95_ang"),
]


def _posneg_mean_from_signed_field(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Return (mean_positive, mean_negative) for a signed field over the sphere.

    Means are computed as integrals over the sphere divided by 4π using the
    Lebedev weights (same convention as other metrics in this app).
    """

    x = np.asarray(x, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    if x.size != w.size:
        raise ValueError("Signed field and weights must have the same length")

    finite = np.isfinite(x) & np.isfinite(w)
    if not np.any(finite):
        return (float("nan"), float("nan"))

    xf = x[finite]
    wf = w[finite]
    pos = float(np.sum(wf * np.maximum(xf, 0.0)) / (4.0 * np.pi))
    neg = float(np.sum(wf * np.minimum(xf, 0.0)) / (4.0 * np.pi))
    return pos, neg


def _metric_plot_value(
    metric_key: str,
    *,
    arrays: Dict[str, np.ndarray],
    metrics: Dict[str, Any],
    w: np.ndarray,
) -> float:
    key = str(metric_key)
    if key in metrics:
        return float(metrics[key])
    if key == "mean_par_rel_signed":
        x = np.asarray(arrays["par_rel_signed"], dtype=float)
        return float(np.sum(w * x) / (4.0 * np.pi))
    if key == "p95_par_rel_signed":
        x = np.asarray(arrays["par_rel_signed"], dtype=float)
        x = x[np.isfinite(x)]
        return float(np.percentile(x, 95)) if x.size else float("nan")
    raise KeyError(f"Unknown plot metric '{metric_key}'")


@lru_cache(maxsize=32)
def _posneg_par_rel_signed_across_bases(
    mol: str,
    omega: float,
    ref_basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
) -> Dict[str, Tuple[float, float]]:
    grid = load_lebedev_grid(int(lebedev_order))
    df = _data()

    beta_ref = _tensor_from_long(df, mol, ref_basis, float(omega))

    R = None
    if coord_system == "principal_axes":
        mol_obj = _load_molecule(mol)
        R = _principal_axes_rotation(mol_obj)
        beta_ref = _rotate_beta(beta_ref, R)

    n_hat = np.asarray(grid.n_hat, dtype=float)
    w = np.asarray(grid.w, dtype=float)
    v_ref = evaluate_field(beta_ref, n_hat)

    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(rel_norm),
    )

    out: Dict[str, Tuple[float, float]] = {}
    for basis in _basis_list():
        family, card = _basis_family_and_cardinal(str(basis))
        if family is None or card is None:
            continue
        if family not in _PLOT_FAMILY_ORDER:
            continue
        if card not in _PLOT_CARD_ORDER:
            continue

        beta_bas = _tensor_from_long(df, mol, str(basis), float(omega))
        if R is not None:
            beta_bas = _rotate_beta(beta_bas, R)
        v_bas = evaluate_field(beta_bas, n_hat)
        arrays, _metrics = compute_error_fields(v_ref, v_bas, n_hat, w, settings=settings)

        pos, neg = _posneg_mean_from_signed_field(arrays["par_rel_signed"], w)
        out[str(basis)] = (pos, neg)

    return out


@lru_cache(maxsize=128)
def _posneg_par_rel_signed_across_omegas_for_basis(
    mol: str,
    ref_basis: str,
    basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
) -> Dict[float, Tuple[float, float]]:
    """Return {omega: (mean_pos, mean_neg)} for a single basis across all omegas."""
    grid = load_lebedev_grid(int(lebedev_order))
    df = _data()

    R = None
    if coord_system == "principal_axes":
        mol_obj = _load_molecule(mol)
        R = _principal_axes_rotation(mol_obj)

    n_hat = np.asarray(grid.n_hat, dtype=float)
    w = np.asarray(grid.w, dtype=float)

    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(rel_norm),
    )

    out: Dict[float, Tuple[float, float]] = {}
    for omega in _omega_list(mol):
        try:
            beta_ref = _tensor_from_long(df, mol, ref_basis, float(omega))
            beta_bas = _tensor_from_long(df, mol, basis, float(omega))
        except KeyError:
            continue
        if R is not None:
            beta_ref = _rotate_beta(beta_ref, R)
            beta_bas = _rotate_beta(beta_bas, R)
        v_ref = evaluate_field(beta_ref, n_hat)
        v_bas = evaluate_field(beta_bas, n_hat)
        arrays, _metrics = compute_error_fields(v_ref, v_bas, n_hat, w, settings=settings)
        out[float(omega)] = _posneg_mean_from_signed_field(arrays["par_rel_signed"], w)

    return out


@lru_cache(maxsize=32)
def _metric_values_across_bases(
    mol: str,
    omega: float,
    ref_basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
    metric_key: str,
) -> Dict[str, float]:
    grid = load_lebedev_grid(int(lebedev_order))
    df = _data()

    beta_ref = _tensor_from_long(df, mol, ref_basis, float(omega))

    R = None
    if coord_system == "principal_axes":
        mol_obj = _load_molecule(mol)
        R = _principal_axes_rotation(mol_obj)
        beta_ref = _rotate_beta(beta_ref, R)

    n_hat = np.asarray(grid.n_hat, dtype=float)
    w = np.asarray(grid.w, dtype=float)
    v_ref = evaluate_field(beta_ref, n_hat)

    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(rel_norm),
    )

    values: Dict[str, float] = {}
    for basis in _basis_list():
        family, card = _basis_family_and_cardinal(str(basis))
        if family is None or card is None:
            continue
        if family not in _PLOT_FAMILY_ORDER:
            continue
        if card not in _PLOT_CARD_ORDER:
            continue
        beta_bas = _tensor_from_long(df, mol, str(basis), float(omega))
        if R is not None:
            beta_bas = _rotate_beta(beta_bas, R)
        v_bas = evaluate_field(beta_bas, n_hat)
        arrays, metrics = compute_error_fields(v_ref, v_bas, n_hat, w, settings=settings)
        values[str(basis)] = _metric_plot_value(metric_key, arrays=arrays, metrics=metrics, w=w)

    return values


@lru_cache(maxsize=128)
def _metric_values_across_omegas_for_basis(
    mol: str,
    ref_basis: str,
    basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
    metric_key: str,
) -> Dict[float, float]:
    """Return {omega: metric_value} for a single basis across all omegas."""
    grid = load_lebedev_grid(int(lebedev_order))
    df = _data()

    R = None
    if coord_system == "principal_axes":
        mol_obj = _load_molecule(mol)
        R = _principal_axes_rotation(mol_obj)

    n_hat = np.asarray(grid.n_hat, dtype=float)
    w = np.asarray(grid.w, dtype=float)

    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(rel_norm),
    )

    out: Dict[float, float] = {}
    for omega in _omega_list(mol):
        try:
            beta_ref = _tensor_from_long(df, mol, ref_basis, float(omega))
            beta_bas = _tensor_from_long(df, mol, basis, float(omega))
        except KeyError:
            continue
        if R is not None:
            beta_ref = _rotate_beta(beta_ref, R)
            beta_bas = _rotate_beta(beta_bas, R)
        v_ref = evaluate_field(beta_ref, n_hat)
        v_bas = evaluate_field(beta_bas, n_hat)
        arrays, metrics = compute_error_fields(v_ref, v_bas, n_hat, w, settings=settings)
        out[float(omega)] = _metric_plot_value(metric_key, arrays=arrays, metrics=metrics, w=w)

    return out


@lru_cache(maxsize=32)
def _metric_plot_data_url(
    mol: str,
    omega: float,
    ref_basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
    metric_key: str,
    plot_x: str,
    omega_agg: str,
) -> str:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    plot_x = str(plot_x)
    omega_agg = str(omega_agg)
    yscale = str(state.plot_yscale)

    metric_key = str(metric_key)

    if plot_x == "omega":
        fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=160)
        omegas = [float(x) for x in _omega_list(mol)]

        # Plot up to 12 lines: each family/cardinal (DZ/TZ/QZ) across omega.
        # For the special pos/neg plot, each basis becomes two lines (+ solid, − dashed).
        bases = _basis_list()
        for fam in _PLOT_FAMILY_ORDER:
            for card in _PLOT_CARD_ORDER:
                # Find the basis string for this family/card.
                basis_name = None
                for b in bases:
                    f, c = _basis_family_and_cardinal(b)
                    if f == fam and c == card:
                        basis_name = b
                        break
                if basis_name is None:
                    continue

                if metric_key == "posneg_par_rel_signed":
                    series = _posneg_par_rel_signed_across_omegas_for_basis(
                        mol,
                        ref_basis,
                        str(basis_name),
                        int(lebedev_order),
                        coord_system,
                        rel_norm,
                    )
                    xs = [w for w in omegas if w in series]
                    ys_pos = [float(series[w][0]) for w in xs]
                    ys_neg = [float(series[w][1]) for w in xs]
                    if not xs:
                        continue
                    ax.plot(xs, ys_pos, marker="o", linewidth=1.5, label=f"{fam}:{card} (+)")
                    ax.plot(
                        xs,
                        ys_neg,
                        marker="o",
                        linewidth=1.5,
                        linestyle="--",
                        label=f"{fam}:{card} (-)",
                    )
                else:
                    series = _metric_values_across_omegas_for_basis(
                        mol,
                        ref_basis,
                        str(basis_name),
                        int(lebedev_order),
                        coord_system,
                        rel_norm,
                        metric_key,
                    )
                    xs = [w for w in omegas if w in series]
                    ys = [float(series[w]) for w in xs]
                    if not xs:
                        continue
                    ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{fam}:{card}")

        ax.set_xlabel("omega")
        ax.set_ylabel("mean(par_rel_signed)+ / mean(par_rel_signed)-" if metric_key == "posneg_par_rel_signed" else str(metric_key))
        ax.grid(True, alpha=0.25)
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-3)
        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()

    else:
        # Default: x-axis is basis set blocks.
        # Optionally show omega-variation as error bars (mean ± std across all ω).

        # Arrange x as blocks: each family gets DZ/TZ/QZ slots.
        x_positions: Dict[Tuple[str, str], float] = {}
        tick_pos: List[float] = []
        tick_lab: List[str] = []
        family_centers: Dict[str, float] = {}
        x = 0.0
        gap = 0.8
        step = 1.0
        for fam in _PLOT_FAMILY_ORDER:
            start = x
            for card in _PLOT_CARD_ORDER:
                x_positions[(fam, card)] = x
                tick_pos.append(x)
                tick_lab.append(card)
                x += step
            end = x - step
            family_centers[fam] = 0.5 * (start + end)
            x += gap

        fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=160)

        bases = _basis_list()
        for fam in _PLOT_FAMILY_ORDER:
            xs: List[float] = []
            ys: List[float] = []
            yerr: List[float] = []

            xs2: List[float] = []
            ys2: List[float] = []
            yerr2: List[float] = []

            for card in _PLOT_CARD_ORDER:
                basis_name = None
                for b in bases:
                    f, c = _basis_family_and_cardinal(b)
                    if f == fam and c == card:
                        basis_name = b
                        break
                if basis_name is None:
                    continue

                x = float(x_positions[(fam, card)])
                if omega_agg == "mean_std":
                    if metric_key == "posneg_par_rel_signed":
                        series = _posneg_par_rel_signed_across_omegas_for_basis(
                            mol,
                            ref_basis,
                            str(basis_name),
                            int(lebedev_order),
                            coord_system,
                            rel_norm,
                        )
                        vals = np.asarray(list(series.values()), dtype=float)
                        if vals.size == 0:
                            continue
                        pos = vals[:, 0]
                        neg = vals[:, 1]
                        pos = pos[np.isfinite(pos)]
                        neg = neg[np.isfinite(neg)]
                        if pos.size == 0 or neg.size == 0:
                            continue
                        y = float(np.mean(pos))
                        e = float(np.std(pos))
                        y2 = float(np.mean(neg))
                        e2 = float(np.std(neg))
                    else:
                        series = _metric_values_across_omegas_for_basis(
                            mol,
                            ref_basis,
                            str(basis_name),
                            int(lebedev_order),
                            coord_system,
                            rel_norm,
                            metric_key,
                        )
                        vals = np.asarray(list(series.values()), dtype=float)
                        vals = vals[np.isfinite(vals)]
                        if vals.size == 0:
                            continue
                        y = float(np.mean(vals))
                        e = float(np.std(vals))
                else:
                    # current ω
                    if metric_key == "posneg_par_rel_signed":
                        values = _posneg_par_rel_signed_across_bases(
                            mol,
                            float(omega),
                            ref_basis,
                            int(lebedev_order),
                            coord_system,
                            rel_norm,
                        )
                        if str(basis_name) not in values:
                            continue
                        y, y2 = values[str(basis_name)]
                        e, e2 = 0.0, 0.0
                    else:
                        values = _metric_values_across_bases(
                            mol,
                            float(omega),
                            ref_basis,
                            int(lebedev_order),
                            coord_system,
                            rel_norm,
                            metric_key,
                        )
                        if str(basis_name) not in values:
                            continue
                        y = float(values[str(basis_name)])
                        e = 0.0

                xs.append(x)
                ys.append(y)
                yerr.append(e)

                if metric_key == "posneg_par_rel_signed":
                    xs2.append(x)
                    ys2.append(float(y2))
                    yerr2.append(float(e2))

            if not xs:
                continue
            order = np.argsort(np.asarray(xs))
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            yerr = [yerr[i] for i in order]

            if metric_key == "posneg_par_rel_signed" and xs2:
                order2 = np.argsort(np.asarray(xs2))
                xs2 = [xs2[i] for i in order2]
                ys2 = [ys2[i] for i in order2]
                yerr2 = [yerr2[i] for i in order2]

            if metric_key == "posneg_par_rel_signed":
                if omega_agg == "mean_std":
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerr,
                        fmt="o-",
                        capsize=3,
                        linewidth=1.5,
                        label=f"{fam} (+)",
                    )
                    ax.errorbar(
                        xs2,
                        ys2,
                        yerr=yerr2,
                        fmt="^-",
                        capsize=3,
                        linewidth=1.5,
                        linestyle="--",
                        label=f"{fam} (-)",
                    )
                else:
                    ax.plot(xs, ys, marker="o", linewidth=1.5, label=f"{fam} (+)")
                    ax.plot(
                        xs2,
                        ys2,
                        marker="^",
                        linewidth=1.5,
                        linestyle="--",
                        label=f"{fam} (-)",
                    )
            else:
                if omega_agg == "mean_std":
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=yerr,
                        fmt="o-",
                        capsize=3,
                        linewidth=1.5,
                        label=fam,
                    )
                else:
                    ax.plot(xs, ys, marker="o", linewidth=1.5, label=fam)

        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lab)
        if tick_pos:
            ax.set_xlim(min(tick_pos) - 0.5, max(tick_pos) + 0.5)
        ax.grid(True, axis="y", alpha=0.25)
        ax.set_ylabel("mean(par_rel_signed)+ / mean(par_rel_signed)-" if metric_key == "posneg_par_rel_signed" else str(metric_key))
        if yscale == "symlog":
            ax.set_yscale("symlog", linthresh=1e-3)

        # Family labels over each block.
        if tick_pos:
            ylim = ax.get_ylim()
            y_text = ylim[1]
            for fam, cx in family_centers.items():
                ax.text(cx, y_text, fam, ha="center", va="bottom", fontsize=8)

        ax.legend(loc="best", fontsize=8)
        fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _triple_metric_plot_data_url(
    mol: str,
    omega: float,
    ref_basis: str,
    lebedev_order: int,
    coord_system: str,
    rel_norm: str,
    *,
    posneg_yscale: str,
) -> str:
    """Return a data URL for a fixed 3-panel summary plot.

    Panels (top -> bottom):
      1) rel_L2 across basis sets (linear)
      2) mean positive / mean negative par_rel_signed across basis sets (linear|symlog)
      3) mean_ang across basis sets (linear)
    """

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""

    # Arrange x as blocks: each family gets DZ/TZ/QZ slots.
    x_positions: Dict[Tuple[str, str], float] = {}
    tick_pos: List[float] = []
    tick_lab: List[str] = []
    family_centers: Dict[str, float] = {}
    x = 0.0
    gap = 0.8
    step = 1.0
    for fam in _PLOT_FAMILY_ORDER:
        start = x
        for card in _PLOT_CARD_ORDER:
            x_positions[(fam, card)] = x
            tick_pos.append(x)
            tick_lab.append(card)
            x += step
        end = x - step
        family_centers[fam] = 0.5 * (start + end)
        x += gap

    fig, (ax1, ax2, ax3) = plt.subplots(
        3,
        1,
        figsize=(9.0, 8.8),
        dpi=160,
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.15, 1.0]},
    )

    bases = _basis_list()

    # 1) rel_L2 (linear)
    values_l2 = _metric_values_across_bases(
        mol,
        float(omega),
        ref_basis,
        int(lebedev_order),
        coord_system,
        rel_norm,
        "rel_L2",
    )
    for fam in _PLOT_FAMILY_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        for card in _PLOT_CARD_ORDER:
            basis_name = None
            for b in bases:
                f, c = _basis_family_and_cardinal(b)
                if f == fam and c == card:
                    basis_name = b
                    break
            if basis_name is None:
                continue
            if str(basis_name) not in values_l2:
                continue
            xs.append(float(x_positions[(fam, card)]))
            ys.append(float(values_l2[str(basis_name)]))
        if xs:
            order = np.argsort(np.asarray(xs))
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax1.plot(xs, ys, marker="o", linewidth=1.5, label=fam)
    ax1.set_ylabel("rel_L2")
    ax1.grid(True, axis="y", alpha=0.25)
    ax1.legend(loc="best", fontsize=8)

    # 2) mean positive / mean negative par_rel_signed (linear|symlog)
    values_posneg = _posneg_par_rel_signed_across_bases(
        mol,
        float(omega),
        ref_basis,
        int(lebedev_order),
        coord_system,
        rel_norm,
    )
    for fam in _PLOT_FAMILY_ORDER:
        xs: List[float] = []
        ys_pos: List[float] = []
        ys_neg: List[float] = []
        for card in _PLOT_CARD_ORDER:
            basis_name = None
            for b in bases:
                f, c = _basis_family_and_cardinal(b)
                if f == fam and c == card:
                    basis_name = b
                    break
            if basis_name is None:
                continue
            if str(basis_name) not in values_posneg:
                continue
            pos, neg = values_posneg[str(basis_name)]
            xs.append(float(x_positions[(fam, card)]))
            ys_pos.append(float(pos))
            ys_neg.append(float(neg))
        if xs:
            order = np.argsort(np.asarray(xs))
            xs = [xs[i] for i in order]
            ys_pos = [ys_pos[i] for i in order]
            ys_neg = [ys_neg[i] for i in order]
            ax2.plot(xs, ys_pos, marker="o", linewidth=1.5, label=f"{fam} (+)")
            ax2.plot(
                xs,
                ys_neg,
                marker="^",
                linewidth=1.5,
                linestyle="--",
                label=f"{fam} (-)",
            )
    ax2.set_ylabel("mean par_rel_signed (+ / -)")
    ax2.grid(True, axis="y", alpha=0.25)
    if str(posneg_yscale) == "symlog":
        ax2.set_yscale("symlog", linthresh=1e-3)
    ax2.legend(loc="best", fontsize=8)

    # 3) mean angle (linear)
    values_ang = _metric_values_across_bases(
        mol,
        float(omega),
        ref_basis,
        int(lebedev_order),
        coord_system,
        rel_norm,
        "mean_ang",
    )
    for fam in _PLOT_FAMILY_ORDER:
        xs: List[float] = []
        ys: List[float] = []
        for card in _PLOT_CARD_ORDER:
            basis_name = None
            for b in bases:
                f, c = _basis_family_and_cardinal(b)
                if f == fam and c == card:
                    basis_name = b
                    break
            if basis_name is None:
                continue
            if str(basis_name) not in values_ang:
                continue
            xs.append(float(x_positions[(fam, card)]))
            ys.append(float(values_ang[str(basis_name)]))
        if xs:
            order = np.argsort(np.asarray(xs))
            xs = [xs[i] for i in order]
            ys = [ys[i] for i in order]
            ax3.plot(xs, ys, marker="o", linewidth=1.5, label=fam)
    ax3.set_ylabel("mean_ang [deg]")
    ax3.grid(True, axis="y", alpha=0.25)
    ax3.legend(loc="best", fontsize=8)

    ax3.set_xticks(tick_pos)
    ax3.set_xticklabels(tick_lab)
    if tick_pos:
        ax3.set_xlim(min(tick_pos) - 0.5, max(tick_pos) + 0.5)

    # Family labels over each block (top plot, for compactness).
    if tick_pos:
        ylim = ax1.get_ylim()
        y_text = ylim[1]
        for fam, cx in family_centers.items():
            ax1.text(cx, y_text, fam, ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    data = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{data}"


def _update_metric_plot() -> None:
    if not _selection_ready():
        state.metric_plot_src = ""
        state.metric_plot_error = ""
        return
    try:
        src = _triple_metric_plot_data_url(
            str(state.mol),
            float(state.omega),
            str(state.ref_basis),
            int(state.lebedev_order),
            str(state.coord_system),
            str(state.rel_norm),
            posneg_yscale=str(state.posneg_yscale),
        )
        state.metric_plot_src = src
        state.metric_plot_error = "" if src else "Plot unavailable (missing matplotlib?)"
    except Exception as exc:
        state.metric_plot_src = ""
        state.metric_plot_error = f"Plot error: {type(exc).__name__}: {exc}"


state.setdefault("mol", "")
state.setdefault("omega", None)
state.setdefault("ref_basis", "")
state.setdefault("bas_basis_a", "")
state.setdefault("bas_basis_b", "")
state.setdefault("lebedev_order", 29)
state.setdefault("field_mode", "bas")
state.setdefault("scalar_mode", "rel_err")
state.setdefault("glyph_factor", 0.12)
state.setdefault("molecule_scale", 0.7)
state.setdefault("atom_radius", 0.25)
state.setdefault("coord_system", "as_is")
state.setdefault("view_layout", "compare_bases")
state.setdefault("scale_by_metric", False)
state.setdefault("rel_norm", "global_rms")
state.setdefault("show_axes", True)
state.setdefault("clim_mode", "auto")
state.setdefault("clim_min", 0.0)
state.setdefault("clim_max", 1.0)
state.setdefault("metrics", {})
state.setdefault("metric_rows", [])
state.setdefault("posneg_yscale", "linear")
state.setdefault("metric_plot_src", "")
state.setdefault("metric_plot_error", "")
state.setdefault(
    "metric_headers",
    [
        {"text": "Metric", "value": "name"},
        {"text": "A", "value": "a"},
        {"text": "B", "value": "b"},
    ],
)


def _set_waiting_status() -> None:
    state.status = "select data"
    state.status_color = "blue"
    state.last_error = ""
    state.metric_rows = []
    state.metric_plot_src = ""
    state.metric_plot_error = ""


def _selection_ready() -> bool:
    mol = str(state.mol).strip()
    if not mol:
        return False
    bases = _basis_list()
    if not bases:
        return False
    if str(state.ref_basis) not in bases:
        return False
    if str(state.bas_basis_a) not in bases:
        return False
    if str(state.bas_basis_b) not in bases:
        return False
    omegas = _omega_list(mol)
    if not omegas:
        return False
    try:
        omega_val = float(state.omega)
    except (TypeError, ValueError):
        return False
    return any(float(o) == omega_val for o in omegas)


@state.change("show_axes")
def _on_show_axes(show_axes, **_):
    vis = bool(show_axes)
    _scene_left["axes_actor"].SetVisibility(1 if vis else 0)
    _scene_mid["axes_actor"].SetVisibility(1 if vis else 0)
    _scene_right["axes_actor"].SetVisibility(1 if vis else 0)
    renderWindow.Modified()
    ctrl.view_update()


def _ensure_basis_selections() -> None:
    bases = _basis_list()
    if not bases:
        return

    if str(state.ref_basis) not in bases:
        state.ref_basis = str(bases[0])

    if str(state.bas_basis_a) not in bases:
        preferred = "aug-cc-pVDZ"
        state.bas_basis_a = preferred if preferred in bases else str(bases[0])

    if str(state.bas_basis_b) not in bases:
        preferred = "d-aug-cc-pVDZ"
        if preferred in bases:
            state.bas_basis_b = preferred
        else:
            fallback = "aug-cc-pVDZ"
            state.bas_basis_b = fallback if fallback in bases else str(bases[0])


def _apply_view_layout() -> None:
    layout = str(state.view_layout)
    if layout == "reference_only":
        _scene_left["renderer"].SetViewport(0.0, 0.0, 1.0, 1.0)
        _scene_mid["renderer"].SetViewport(0.0, 0.0, 0.0, 0.0)
        _scene_right["renderer"].SetViewport(0.0, 0.0, 0.0, 0.0)
        _scene_mid["scalar_bar"].VisibilityOff()
        _scene_right["scalar_bar"].VisibilityOff()
    elif layout == "basis_only":
        _scene_left["renderer"].SetViewport(0.0, 0.0, 0.0, 0.0)
        _scene_left["scalar_bar"].VisibilityOff()
        _scene_mid["renderer"].SetViewport(0.0, 0.0, 1.0, 1.0)
        _scene_right["renderer"].SetViewport(0.0, 0.0, 0.0, 0.0)
        _scene_right["scalar_bar"].VisibilityOff()
    elif layout == "compare_bases":
        _scene_left["renderer"].SetViewport(0.0, 0.0, 1.0 / 3.0, 1.0)
        _scene_mid["renderer"].SetViewport(1.0 / 3.0, 0.0, 2.0 / 3.0, 1.0)
        _scene_right["renderer"].SetViewport(2.0 / 3.0, 0.0, 1.0, 1.0)
    else:
        # side_by_side: reference + basis A
        _scene_left["renderer"].SetViewport(0.0, 0.0, 0.5, 1.0)
        _scene_mid["renderer"].SetViewport(0.5, 0.0, 1.0, 1.0)
        _scene_right["renderer"].SetViewport(0.0, 0.0, 0.0, 0.0)
        _scene_right["scalar_bar"].VisibilityOff()


@state.change("view_layout")
def _on_layout_change(**_):
    _apply_view_layout()
    _update_corner_annotations()
    renderWindow.Modified()
    ctrl.view_update()


def _clear_scene(scene: Dict[str, Any]) -> None:
    renderer = scene["renderer"]
    if scene.get("glyph_actor") is not None:
        renderer.RemoveActor(scene["glyph_actor"])
        scene["glyph_actor"] = None
    for a in scene.get("mol_actors", []):
        renderer.RemoveActor(a)
    scene["mol_actors"] = []


def _set_scalar_bar(
    scene: Dict[str, Any],
    *,
    actor: vtkActor,
    title: str,
    mode: str,
    clim: Tuple[float, float],
) -> None:
    scalar_bar = scene["scalar_bar"]
    glyph_mapper = actor.GetMapper()
    lut = glyph_mapper.GetLookupTable()
    scalar_bar.SetLookupTable(lut)

    scalar_bar.SetTitle(str(title))

    # Standard scalar bar behavior.
    scalar_bar.SetUseCustomLabels(False)
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLabelFormat("%-#6.3g")

    scalar_bar.VisibilityOn()


def _coord_label() -> str:
    cs = str(state.coord_system)
    if cs == "principal_axes":
        return "principal_axes"
    return "as_is"


def _layout_label() -> str:
    layout = str(state.view_layout)
    if layout == "reference_only":
        return "ref_only"
    if layout == "basis_only":
        return "basis_only"
    if layout == "compare_bases":
        return "compare"
    return "side_by_side"


def _update_corner_annotations() -> None:
    # Titles: upper-left. CornerAnnotation is reliably supported in VTK.js.
    _scene_left["corner"].SetText(2, f"Reference ({state.ref_basis})")
    _scene_mid["corner"].SetText(2, f"Basis A ({state.bas_basis_a})")
    _scene_right["corner"].SetText(2, f"Basis B ({state.bas_basis_b})")

    # Context: compact upper-right.
    base = f"Coord: {_coord_label()}  |  View: {_layout_label()}  |  RelNorm: {_rel_norm_label()}"
    _scene_left["corner"].SetText(3, base)
    _scene_mid["corner"].SetText(3, base)
    _scene_right["corner"].SetText(3, base)

    # Best-effort: also set TextActor titles (may not render in all VTK.js builds).
    if _scene_left.get("title_actor") is not None:
        _scene_left["title_actor"].SetInput(f"Reference ({state.ref_basis})")
    if _scene_mid.get("title_actor") is not None:
        _scene_mid["title_actor"].SetInput(f"Basis A ({state.bas_basis_a})")
    if _scene_right.get("title_actor") is not None:
        _scene_right["title_actor"].SetInput(f"Basis B ({state.bas_basis_b})")


def _compute_bundle():
    grid = load_lebedev_grid(int(state.lebedev_order))
    df = _data()

    beta_ref = _tensor_from_long(df, state.mol, state.ref_basis, float(state.omega))
    beta_a = _tensor_from_long(df, state.mol, state.bas_basis_a, float(state.omega))
    beta_b = _tensor_from_long(df, state.mol, state.bas_basis_b, float(state.omega))

    R = None
    if state.coord_system == "principal_axes":
        mol = _load_molecule(state.mol)
        R = _principal_axes_rotation(mol)
        beta_ref = _rotate_beta(beta_ref, R)
        beta_a = _rotate_beta(beta_a, R)
        beta_b = _rotate_beta(beta_b, R)

    # Support both: LebedevGrid with attributes (.n_hat/.w) and legacy tuple (n_hat, w)
    if isinstance(grid, tuple) and len(grid) == 2:
        n_hat = np.asarray(grid[0], dtype=float)
        w = np.asarray(grid[1], dtype=float)
    else:
        n_hat = np.asarray(grid.n_hat, dtype=float)
        w = np.asarray(grid.w, dtype=float)

    v_ref = evaluate_field(beta_ref, n_hat)
    v_a = evaluate_field(beta_a, n_hat)
    v_b = evaluate_field(beta_b, n_hat)

    settings = ErrorSettings(
        enable_mask=True,
        mask_threshold_fraction=0.01,
        component_metrics=True,
        radial_tangential_metrics=True,
        rel_norm=str(state.rel_norm),
    )

    arrays_a, metrics_a = compute_error_fields(v_ref, v_a, n_hat, w, settings=settings)
    arrays_b, metrics_b = compute_error_fields(v_ref, v_b, n_hat, w, settings=settings)

    return n_hat, w, v_ref, v_a, v_b, arrays_a, arrays_b, metrics_a, metrics_b, R


def _rebuild_scene(
    scene: Dict[str, Any],
    *,
    n_hat: np.ndarray,
    vectors: np.ndarray,
    scalars: np.ndarray,
    scale: np.ndarray,
    clim: Tuple[float, float],
    scalar_title: str,
    scalar_mode: str,
    lut_kind: str,
    R: np.ndarray | None,
 ) -> None:
    renderer = scene["renderer"]
    renderWindow = scene["renderWindow"]
    _clear_scene(scene)

    vectors = np.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    scalars = np.nan_to_num(scalars, nan=0.0, posinf=0.0, neginf=0.0)
    scale = np.nan_to_num(scale, nan=0.0, posinf=0.0, neginf=0.0)

    glyph_actor = _build_glyph_actor(
        n_hat=n_hat,
        vectors=vectors,
        scalars=scalars,
        scale=scale,
        glyph_factor=float(state.glyph_factor),
        clim=clim,
        lut_kind=lut_kind,
        scalar_mode=str(scalar_mode),
    )
    renderer.AddActor(glyph_actor)
    scene["glyph_actor"] = glyph_actor
    _set_scalar_bar(scene, actor=glyph_actor, title=scalar_title, mode=str(scalar_mode), clim=clim)

    mol = _load_molecule(state.mol)
    symbols = list(mol.symbols)
    coords = np.asarray(mol.geometry, dtype=float)
    coords = coords - np.mean(coords, axis=0, keepdims=True)
    if R is not None:
        coords = coords @ R
    max_norm = float(np.max(np.linalg.norm(coords, axis=1))) if coords.size else 1.0
    if max_norm > 0:
        coords = coords / max_norm
    coords = coords * float(state.molecule_scale)

    for sym in sorted(set(symbols)):
        idx = [i for i, s in enumerate(symbols) if s == sym]
        pts = coords[idx]
        actor = _build_atom_actor(pts, radius=float(state.atom_radius), rgb=_element_rgb(sym))
        renderer.AddActor(actor)
        scene["mol_actors"].append(actor)

    # With VTK.js client-side rendering, we do not need server-side OpenGL renders.
    # Avoid calling Render() to prevent headless/OpenGL stalls.


def _rebuild_both() -> None:
    import traceback
    global _camera_initialized

    try:
        if not _selection_ready():
            _set_waiting_status()
            _scene_left["scalar_bar"].VisibilityOff()
            _scene_mid["scalar_bar"].VisibilityOff()
            _scene_right["scalar_bar"].VisibilityOff()
            ctrl.view_update()
            return
        print("[trame] rebuilding scenes ...", flush=True)
        state.status = "building scene"
        state.status_color = "orange"
        state.last_error = ""

        # Apply the current layout (important for initial load).
        _apply_view_layout()

        n_hat, w, v_ref, v_a, v_b, arrays_a, arrays_b, metrics_a, metrics_b, R = _compute_bundle()
        _update_corner_annotations()

        # LEFT: fixed to reference magnitude, colored by |v_ref|
        left_vectors = v_ref
        left_scalars = arrays_a["ref_mag"]
        left_scale = arrays_a["ref_mag"]
        left_clim = _auto_clim(left_scalars, mode="ref_mag")
        _rebuild_scene(
            _scene_left,
            n_hat=n_hat,
            vectors=left_vectors,
            scalars=left_scalars,
            scale=left_scale,
            clim=left_clim,
            scalar_title=_scalar_mode_label("ref_mag"),
            scalar_mode="ref_mag",
            lut_kind="sequential",
            R=R,
        )

        def _vectors_and_scale(field_mode: str, *, v_bas: np.ndarray, arrays: Dict[str, np.ndarray]):
            if field_mode == "ref":
                return v_ref, arrays["ref_mag"]
            if field_mode == "bas":
                return v_bas, arrays["bas_mag"]
            return arrays["dv"], arrays["abs_err"]

        mode = str(state.scalar_mode)
        if mode not in arrays_a:
            raise KeyError(
                f"Unknown scalar_mode={mode}. Available: {sorted(arrays_a.keys())}"
            )
        style = _metric_style(mode)
        lut_kind = "diverging" if style["signed"] else "sequential"

        global_clims = None
        if str(state.clim_mode) == "auto":
            global_clims = _global_metric_clims(
                str(state.mol),
                float(state.omega),
                str(state.ref_basis),
                int(state.lebedev_order),
                str(state.coord_system),
                str(state.rel_norm),
            )

        # BASIS A scene
        a_vectors, a_scale = _vectors_and_scale(str(state.field_mode), v_bas=v_a, arrays=arrays_a)
        a_scalars = arrays_a[mode]
        if state.clim_mode == "manual":
            a_clim = (float(state.clim_min), float(state.clim_max))
        else:
            a_clim = (
                global_clims.get(mode)  # type: ignore[union-attr]
                if global_clims is not None and mode in global_clims
                else _auto_clim(a_scalars, mode=mode)
            )
        if bool(state.scale_by_metric):
            a_scale = np.abs(a_scalars) if style["signed"] else a_scalars
        _rebuild_scene(
            _scene_mid,
            n_hat=n_hat,
            vectors=a_vectors,
            scalars=a_scalars,
            scale=a_scale,
            clim=a_clim,
            scalar_title=_scalar_mode_label(mode),
            scalar_mode=mode,
            lut_kind=lut_kind,
            R=R,
        )

        # BASIS B scene
        b_vectors, b_scale = _vectors_and_scale(str(state.field_mode), v_bas=v_b, arrays=arrays_b)
        b_scalars = arrays_b[mode]
        if state.clim_mode == "manual":
            b_clim = (float(state.clim_min), float(state.clim_max))
        else:
            b_clim = (
                global_clims.get(mode)  # type: ignore[union-attr]
                if global_clims is not None and mode in global_clims
                else _auto_clim(b_scalars, mode=mode)
            )
        if bool(state.scale_by_metric):
            b_scale = np.abs(b_scalars) if style["signed"] else b_scalars
        _rebuild_scene(
            _scene_right,
            n_hat=n_hat,
            vectors=b_vectors,
            scalars=b_scalars,
            scale=b_scale,
            clim=b_clim,
            scalar_title=_scalar_mode_label(mode),
            scalar_mode=mode,
            lut_kind=lut_kind,
            R=R,
        )

        # One shared camera for all renderers.
        if not _camera_initialized:
            _scene_left["renderer"].ResetCamera()
            _camera_initialized = True

        metric_keys = (
            "rel_L2",
            "L2_par",
            "L2_perp",
            "ratio_perp_par",
            "max_rel",
            "p95_rel",
            "mean_ang",
            "p95_ang",
            "bias_ratio",
            "masked_rel_L2",
        )
        state.metric_rows = [
            {"name": k, "a": metrics_a.get(k), "b": metrics_b.get(k)} for k in metric_keys
        ]
        state.metric_headers = [
            {"text": "Metric", "value": "name"},
            {"text": f"A: {state.bas_basis_a}", "value": "a"},
            {"text": f"B: {state.bas_basis_b}", "value": "b"},
        ]

        _update_metric_plot()

        print("[trame] scenes rebuilt", flush=True)
        state.status = "ready"
        state.status_color = "green"
        renderWindow.Modified()
        ctrl.view_update()

    except Exception as exc:
        tb = traceback.format_exc()
        print("[trame] rebuild failed:\n" + tb, flush=True)
        state.last_error = f"{type(exc).__name__}: {exc}"
        state.status = "error"
        state.status_color = "red"
        try:
            _scene_left["scalar_bar"].VisibilityOff()
            _scene_mid["scalar_bar"].VisibilityOff()
            _scene_right["scalar_bar"].VisibilityOff()
            renderWindow.Modified()
            ctrl.view_update()
        except Exception:
            pass


@state.change("mol")
def _on_mol_change(mol, **_):
    global _camera_initialized
    if not str(mol).strip():
        state.omega = None
        _set_waiting_status()
        ctrl.view_update()
        return
    omegas = _omega_list(mol)
    if omegas:
        try:
            omega_val = float(state.omega)
        except (TypeError, ValueError):
            omega_val = None
        if omega_val is None or omega_val not in omegas:
            state.omega = float(omegas[0])
    _ensure_basis_selections()
    _camera_initialized = False
    if _selection_ready():
        _rebuild_both()
    else:
        _set_waiting_status()


@state.change(
    "omega",
    "ref_basis",
    "bas_basis_a",
    "bas_basis_b",
    "lebedev_order",
    "field_mode",
    "scalar_mode",
    "glyph_factor",
    "molecule_scale",
    "atom_radius",
    "coord_system",
    "rel_norm",
    "scale_by_metric",
    "clim_mode",
    "clim_min",
    "clim_max",
)
def _on_params_change(**_):
    if _selection_ready():
        _rebuild_both()
    else:
        _set_waiting_status()


@state.change("posneg_yscale")
def _on_posneg_yscale_change(**_):
    _update_metric_plot()
    renderWindow.Modified()
    ctrl.view_update()


def standard_buttons():
    vuetify.VCheckbox(
        v_model="$vuetify.theme.dark",
        on_icon="mdi-lightbulb-off-outline",
        off_icon="mdi-lightbulb-outline",
        classes="mx-1",
        hide_details=True,
        dense=True,
    )
    with vuetify.VBtn(icon=True, click=ctrl.reset_camera):
        vuetify.VIcon("mdi-crop-free")


def _metrics_table():
    vuetify.VDataTable(
        dense=True,
        disable_pagination=True,
        hide_default_footer=True,
        items=("metric_rows", []),
        headers=("metric_headers", []),
    )


with SinglePageWithDrawerLayout(server) as layout:
    layout.title.set_text("β unit-sphere viewer")

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VDivider(vertical=True, classes="mx-2")
        vuetify.VChip(
            "{{ status }}",
            small=True,
            outlined=True,
            color=("status_color", "orange"),
            classes="mr-2",
        )
        standard_buttons()

    with layout.drawer as drawer:
        drawer.width = 360
        vuetify.VContainer(fluid=True, classes="pa-2")

        vuetify.VAlert(
            "{{ last_error }}",
            type="error",
            dense=True,
            outlined=True,
            prominent=False,
            v_if="last_error",
            classes="mb-2",
        )

        vuetify.VSelect(
            label="Molecule",
            v_model=("mol", state.mol),
            items=("molecule_items", _molecule_list()),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VSelect(
            label="Coordinate system",
            v_model=("coord_system", state.coord_system),
            items=(
                "coord_items",
                [
                    {"text": "As-is (tensor frame)", "value": "as_is"},
                    {"text": "Principal axes (inertia)", "value": "principal_axes"},
                ],
            ),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VCheckbox(
            label="Show axes",
            v_model=("show_axes", state.show_axes),
            dense=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VSelect(
            label="View",
            v_model=("view_layout", state.view_layout),
            items=(
                "view_items",
                [
                    {"text": "Side-by-side", "value": "side_by_side"},
                    {"text": "Compare bases (A vs B)", "value": "compare_bases"},
                    {"text": "Reference only", "value": "reference_only"},
                    {"text": "Basis/errors only", "value": "basis_only"},
                ],
            ),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Omega",
            v_model=("omega", state.omega),
            items=("omega_items", _omega_list(state.mol)),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Reference basis",
            v_model=("ref_basis", state.ref_basis),
            items=("basis_items", _basis_list()),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Basis A",
            v_model=("bas_basis_a", state.bas_basis_a),
            items=("basis_items", _basis_list()),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Basis B",
            v_model=("bas_basis_b", state.bas_basis_b),
            items=("basis_items", _basis_list()),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Lebedev order",
            v_model=("lebedev_order", state.lebedev_order),
            items=("lebedev_items", [3, 5, 7, 9, 11, 13, 15, 17,
                19, 21, 23, 25, 27, 29, 31, 35,
                41, 47, 53, 59, 65, 71, 77, 83,
                89, 95, 101, 107, 113, 119, 125, 131]),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Vector field",
            v_model=("field_mode", state.field_mode),
            items=("field_items", [{"text": t, "value": v} for t, v in FIELD_CHOICES]),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSelect(
            label="Color by",
            v_model=("scalar_mode", state.scalar_mode),
            items=("metric_items", [{"text": t, "value": v} for t, v in METRIC_CHOICES]),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VSelect(
            label="Relative normalization",
            v_model=("rel_norm", state.rel_norm),
            items=(
                "rel_norm_items",
                [
                    {"text": "Global RMS (recommended)", "value": "global_rms"},
                    {"text": "Global max", "value": "global_max"},
                    {"text": "Global mean", "value": "global_mean"},
                    {"text": "Pointwise |v_ref(n)| (old)", "value": "pointwise"},
                ],
            ),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VCheckbox(
            label="Scale arrows by color metric",
            v_model=("scale_by_metric", state.scale_by_metric),
            dense=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VSelect(
            label="Color limits (basis/errors)",
            v_model=("clim_mode", state.clim_mode),
            items=(
                "clim_mode_items",
                [
                    {"text": "Auto", "value": "auto"},
                    {"text": "Manual", "value": "manual"},
                ],
            ),
            dense=True,
            outlined=True,
            hide_details=True,
            classes="mb-2",
        )

        vuetify.VTextField(
            label="Min",
            v_model=("clim_min", state.clim_min),
            type="number",
            dense=True,
            outlined=True,
            hide_details=True,
            disabled=("clim_mode !== 'manual'",),
            classes="mb-2",
        )
        vuetify.VTextField(
            label="Max",
            v_model=("clim_max", state.clim_max),
            type="number",
            dense=True,
            outlined=True,
            hide_details=True,
            disabled=("clim_mode !== 'manual'",),
            classes="mb-2",
        )

        vuetify.VSlider(
            label="Arrow scale",
            v_model=("glyph_factor", state.glyph_factor),
            min=0.01,
            max=0.5,
            step=0.01,
            dense=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSlider(
            label="Molecule scale",
            v_model=("molecule_scale", state.molecule_scale),
            min=0.2,
            max=1.5,
            step=0.05,
            dense=True,
            hide_details=True,
            classes="mb-2",
        )
        vuetify.VSlider(
            label="Atom radius",
            v_model=("atom_radius", state.atom_radius),
            min=0.05,
            max=0.6,
            step=0.01,
            dense=True,
            hide_details=True,
            classes="mb-4",
        )

        vuetify.VDivider(classes="my-2")
        vuetify.VSubheader("Global metrics")
        _metrics_table()

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            with vuetify.VRow(no_gutters=True, classes="fill-height"):
                with vuetify.VCol(cols=8, classes="pa-0 fill-height"):
                    # Use client-side VTK.js rendering to avoid server OpenGL/EGL issues.
                    # Left/right are renderers in a shared renderWindow, with a shared camera.
                    view = vtk.VtkLocalView(
                        renderWindow,
                        ref="view",
                        style="width: 100%; height: 100%;",
                    )
                    ctrl.view_update = view.update

                with vuetify.VCol(cols=4, classes="pa-2", style="height: 100%; overflow-y: auto;"):
                    vuetify.VSubheader("Metric plots")
                    vuetify.VSelect(
                        label="Signed pos/neg y-scale",
                        v_model=("posneg_yscale", state.posneg_yscale),
                        items=(
                            "posneg_yscale_items",
                            [
                                {"text": "Linear", "value": "linear"},
                                {"text": "SymLog", "value": "symlog"},
                            ],
                        ),
                        dense=True,
                        outlined=True,
                        hide_details=True,
                        classes="mb-2",
                    )
                    vuetify.VAlert(
                        "{{ metric_plot_error }}",
                        type="warning",
                        dense=True,
                        outlined=True,
                        prominent=False,
                        v_if="metric_plot_error",
                        classes="mb-2",
                    )
                    vuetify.VImg(
                        src=("metric_plot_src", ""),
                        contain=True,
                        v_if="metric_plot_src",
                        classes="mb-2",
                        style="width: 100%;",
                    )

            # Now that we have live views, build the initial scenes.
            if _selection_ready():
                _rebuild_both()
            else:
                _set_waiting_status()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    args = _CLI_ARGS or _parse_args()
    kwargs: Dict[str, Any] = {}
    host = args.host
    port = _select_free_port(host, args.port)
    kwargs["host"] = host
    kwargs["port"] = port

    url_host = host
    if url_host in ("0.0.0.0", "::"):
        url_host = "127.0.0.1"
    print(f"Trame server running at http://{url_host}:{port}/")
    server.start(**kwargs)
