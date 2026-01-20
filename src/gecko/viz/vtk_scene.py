from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonCore import vtkLookupTable, vtkPoints
from vtkmodules.vtkCommonDataModel import vtkPolyData
from vtkmodules.vtkFiltersCore import vtkGlyph3D
from vtkmodules.vtkFiltersSources import vtkArrowSource, vtkSphereSource
from vtkmodules.vtkRenderingAnnotation import vtkAxesActor, vtkCornerAnnotation, vtkScalarBarActor
from vtkmodules.vtkRenderingCore import (
    vtkActor,
    vtkCamera,
    vtkColorTransferFunction,
    vtkLogLookupTable,
    vtkPolyDataMapper,
    vtkRenderer,
    vtkRenderWindow,
    vtkRenderWindowInteractor,
    vtkTextActor,
)

# Rendering backend import (safe to include)
import vtkmodules.vtkRenderingOpenGL2  # noqa


def element_rgb(symbol: str) -> Tuple[float, float, float]:
    table = {
        "H": (0.95, 0.95, 0.95),
        "C": (0.2, 0.2, 0.2),
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


def create_render_window(*, debug_fn=None) -> vtkRenderWindow:
    import os

    def _dprint(msg: str) -> None:
        if debug_fn is not None:
            debug_fn(msg)

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

    rw = vtkRenderWindow()
    _dprint("[startup] using default vtkRenderWindow")
    return rw


def new_scene(
    *,
    renderWindow: vtkRenderWindow,
    viewport: Tuple[float, float, float, float],
    background: Tuple[float, float, float] = (0.08, 0.08, 0.10),
) -> Dict[str, Any]:
    renderer = vtkRenderer()
    renderer.SetBackground(float(background[0]), float(background[1]), float(background[2]))
    renderer.SetViewport(float(viewport[0]), float(viewport[1]), float(viewport[2]), float(viewport[3]))
    renderWindow.AddRenderer(renderer)

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

    axes_actor = vtkAxesActor()
    axes_actor.SetTotalLength(0.55, 0.55, 0.55)
    axes_actor.SetShaftTypeToCylinder()
    axes_actor.SetCylinderRadius(0.02)
    axes_actor.AxisLabelsOn()
    renderer.AddActor(axes_actor)

    scalar_bar = vtkScalarBarActor()
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetLabelFormat("%-#6.3g")
    scalar_bar.SetPosition(0.83, 0.10)
    scalar_bar.SetPosition2(0.15, 0.80)
    scalar_bar.VisibilityOff()
    renderer.AddViewProp(scalar_bar)

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


def polydata_from_points(points: np.ndarray) -> vtkPolyData:
    points = np.asarray(points, dtype=float)
    vtk_pts = vtkPoints()
    vtk_pts.SetData(numpy_to_vtk(points, deep=True))
    pd = vtkPolyData()
    pd.SetPoints(vtk_pts)
    return pd


def set_vectors(pd: vtkPolyData, name: str, vectors: np.ndarray) -> None:
    vectors = np.asarray(vectors, dtype=float)
    vtk_arr = numpy_to_vtk(vectors, deep=True)
    vtk_arr.SetName(name)
    vtk_arr.SetNumberOfComponents(3)
    pd.GetPointData().SetVectors(vtk_arr)


def set_scalars(pd: vtkPolyData, name: str, scalars: np.ndarray) -> None:
    scalars = np.asarray(scalars, dtype=float).reshape(-1)
    vtk_arr = numpy_to_vtk(scalars, deep=True)
    vtk_arr.SetName(name)
    vtk_arr.SetNumberOfComponents(1)
    pd.GetPointData().SetScalars(vtk_arr)


def add_array(pd: vtkPolyData, name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    vtk_arr = numpy_to_vtk(arr, deep=True)
    vtk_arr.SetName(name)
    pd.GetPointData().AddArray(vtk_arr)


def build_lut(*, kind: str, clim: Tuple[float, float], scalar_mode: str) -> vtkLookupTable:
    vmin, vmax = float(clim[0]), float(clim[1])
    if vmin == vmax:
        vmax = vmin + 1.0

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

    lut = vtkLookupTable()
    lut.SetRange(vmin, vmax)
    lut.SetNumberOfTableValues(256)
    lut.SetRampToLinear()
    for i in range(256):
        x = vmin + (i / 255.0) * (vmax - vmin)
        r, g, b = ctf.GetColor(x)
        lut.SetTableValue(i, float(r), float(g), float(b), 1.0)
    lut.Build()

    if scalar_mode in ("dv_par_scalar", "par_rel_signed"):
        log_lut = vtkLogLookupTable()
        log_lut.DeepCopy(lut)
        log_lut.SetRange(vmin, vmax)
        log_lut.Build()
        return log_lut

    return lut


def build_glyph_actor(
    points: np.ndarray,
    vectors: np.ndarray,
    scalars: np.ndarray,
    *,
    scale_factor: float,
    lut: vtkLookupTable,
) -> vtkActor:
    pd = polydata_from_points(points)
    set_vectors(pd, "vectors", vectors)
    set_scalars(pd, "scalars", scalars)
    add_array(pd, "scale", scalars)

    arrow = vtkArrowSource()
    arrow.SetTipResolution(24)
    arrow.SetShaftResolution(24)

    glyph = vtkGlyph3D()
    glyph.SetInputData(pd)
    glyph.SetSourceConnection(arrow.GetOutputPort())
    glyph.SetVectorModeToUseVector()
    glyph.SetScaleModeToScaleByScalar()
    glyph.OrientOn()
    glyph.SetScaleFactor(float(scale_factor))

    mapper = vtkPolyDataMapper()
    mapper.SetInputConnection(glyph.GetOutputPort())
    mapper.SetLookupTable(lut)
    mapper.SetColorModeToMapScalars()
    mapper.SelectColorArray("scalars")
    mapper.SetScalarModeToUsePointData()
    mapper.SetUseLookupTableScalarRange(True)

    actor = vtkActor()
    actor.SetMapper(mapper)
    return actor


def build_atom_actor(points: np.ndarray, *, radius: float, rgb: Tuple[float, float, float]) -> vtkActor:
    pd = polydata_from_points(points)
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
