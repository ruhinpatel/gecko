# Trame + VTK guide (for this repo)

This document explains the moving parts that make the interactive unit-sphere viewer work, and how to extend it safely.

It is written specifically for the current app implementation in:
- [notebooks/scripts/application.py](notebooks/scripts/application.py)

## Mental model

There are three layers that work together:

1. **Data & math layer (NumPy / Pandas)**
   - Loads SHG tensors from CSV
   - Builds the unit-sphere sample grid (Lebedev)
   - Evaluates vector fields on the sphere
   - Computes error fields and global scalar metrics

2. **VTK scene layer (vtkmodules)**
   - Converts NumPy arrays into `vtkPolyData`
   - Builds geometry pipelines (arrows for vectors, spheres for atoms)
   - Applies color maps and scalar bars
   - Maintains renderers / camera(s)

3. **Trame UI layer (trame + Vuetify + trame-vtk)**
   - Defines reactive UI controls (dropdowns, sliders)
   - Watches state changes and triggers a scene rebuild
   - Uses **client-side VTK.js rendering** (`VtkLocalView`) so the browser does the actual OpenGL rendering

A helpful way to think of it:

- Python builds a *scene description* (VTK pipeline objects + arrays).
- Trame serializes that scene to JSON-friendly payloads.
- The browser renders it with VTK.js.

## Files and responsibilities

### Core viewer

- [notebooks/scripts/application.py](notebooks/scripts/application.py)
  - Defines the Trame server and Vuetify UI
  - Owns the VTK pipeline builders (glyphs, atoms, scalar bar)
  - Bridges UI state to data computation and VTK scene updates

### Field evaluation and error metrics

- [field_error.py](field_error.py)
  - `load_lebedev_grid(order)` provides unit-sphere directions and weights
  - `evaluate_field(beta, n_hat)` maps a 3×3×3 tensor onto vectors on the sphere
  - `compute_error_fields(v_ref, v_bas, n_hat, w, settings=...)` produces:
    - per-point arrays (magnitudes, abs/rel error, angle error, parallel/perp decomposition, etc.)
    - weighted global metrics (L2-like, percentiles, masked metrics)

This separation is intentional:
- `field_error.py` is the *single source of truth* for the math.
- The viewer imports and uses it so plots and computed metrics stay consistent.

## Trame concepts used in the app

### Server, state, controller

In Trame:
- `server = get_server(...)` creates the app instance.
- `state` is a reactive key/value store.
- `ctrl` is a place to hang callable hooks (e.g., view updates).

In this repo you’ll see patterns like:
- `state.setdefault("mol", "H2O")` for defaults
- `@state.change("mol")` to rebuild when a key changes

Why this matters:
- Trame automatically pushes state changes to the browser.
- The Python callbacks are where you *decide what to rebuild*.

### UI widgets (Vuetify)

The drawer controls are standard Vuetify widgets:
- `VSelect` → dropdowns
- `VSlider` → continuous values
- `VDataTable` → metrics table

Each binds to `state` through `v_model=("some_key", state.some_key)`.

When you change a dropdown, Trame updates `state.some_key`, and any matching `@state.change(...)` callback runs.

### VTK view widget (client-side)

The rendering widget is:
- `vtk.VtkLocalView(renderWindow, ...)`

Key point:
- With **VtkLocalView**, the browser renders using VTK.js.
- That’s why you can run this on a machine with sketchy server-side OpenGL support.

Practically:
- You generally do *not* need to call `Render()` frequently on the server.
- You *do* need to call the view’s update hook after changing the VTK pipeline, so the browser receives the new serialized state.

In this app:
- `ctrl.view_update = view.update`
- `ctrl.view_update()` is called after a rebuild.

## VTK concepts used in the app

### The three essential VTK objects

- `vtkRenderer`: holds actors, background color, camera
- `vtkRenderWindow`: holds one or more renderers
- `vtkActor`: a renderable thing (geometry + properties)

The app uses **two renderers** inside one render window:
- left viewport: reference view
- right viewport: basis/error view

Both renderers share **one camera** so interaction stays linked.

### Converting NumPy arrays into VTK

VTK does not understand NumPy arrays directly.

The conversion flow is:

1. Create a `vtkPolyData` with points (the Lebedev `n_hat` directions)
2. Attach arrays to `PointData`:
   - vectors (`SetVectors`) for arrow orientation
   - scalars for coloring
   - a separate scalar array for glyph sizing

In code this is done by small helpers:
- `_polydata_from_points(points)`
- `_set_vectors(pd, name, vectors)`
- `_set_scalars(pd, name, scalars)`
- `_add_array(pd, name, arr)`

The reason for multiple scalar arrays:
- VTK often expects one “active scalars” array.
- Here we need **one array for color** and **one array for glyph scale**.

### Glyphs (vector arrows)

To draw arrows at each sphere direction:

- Build a point cloud (`vtkPolyData`) of directions
- Add a vector array (arrow direction)
- Use `vtkGlyph3D` with a `vtkArrowSource` as the glyph geometry

Pipeline:
- `vtkArrowSource` → provides arrow mesh
- `vtkGlyph3D` → copies that mesh to each point, oriented by the vectors
- `vtkPolyDataMapper` → turns glyph output into something renderable
- `vtkActor` → holds the mapper and properties

Why `vtkGlyph3D` (not `vtkGlyph3DMapper`):
- Client-side VTK.js serialization tends to behave more reliably with the classic filter + mapper pipeline.

### Coloring + scalar bar

Coloring is done by:
- Creating a `vtkColorTransferFunction` (a LUT)
- Setting LUT range (`SetRange`) and points
- Mapping the *color scalar array* via:
  - `mapper.SetScalarModeToUsePointFieldData()`
  - `mapper.SelectColorArray("scalars")`

The scalar bar (`vtkScalarBarActor`) is then pointed at the mapper’s LUT.

If you add a new metric:
- you need to compute a new `arrays["your_metric"]`
- choose its color range (`clim`)
- ensure the scalar bar title matches your metric label

### Molecule atoms

Atoms are drawn as spheres at atomic coordinates.

Flow:
- Load molecule geometry (repo’s `.mol` files)
- Center and normalize it to fit inside the unit sphere
- For each element type, build a glyph actor with sphere source

This is a typical VTK trick:
- represent atoms as points
- use `vtkGlyph3D` to stamp a sphere mesh at each point

## How the rebuild loop works

There are two phases:

1. **Compute phase** (`_compute_bundle()`)
   - Load tensors for the selected molecule/basis/omega
   - Build the Lebedev grid
   - Compute `v_ref`, `v_bas` and the error `arrays`

2. **Scene phase** (`_rebuild_both()`)
   - Decide what the left view shows (fixed: `|v_ref|`)
   - Decide what the right view shows (interactive)
   - Build glyph actors + atom actors into each renderer
   - Trigger `ctrl.view_update()` so the browser re-serializes and re-renders

Important performance note:
- It’s usually worth caching CSV and molecule loading.
- It’s usually *not* worth caching the final VTK glyph output unless you have very large grids.

## Extending the app

### Add a new per-point scalar metric

1. Add it to the math layer (preferred):
   - Implement in [field_error.py](field_error.py) inside `compute_error_fields` so it’s available everywhere.

2. Wire it into the viewer:
   - Add it to `METRIC_CHOICES` in [notebooks/scripts/application.py](notebooks/scripts/application.py)
   - Ensure `_rebuild_both()` can find it in `arrays`
   - Decide whether it should use symmetric color limits (diverging) or standard limits

### Add a new “vector field” option

Options are controlled by `FIELD_CHOICES` and by selection logic in `_rebuild_both()`.

If you add a new vector field:
- compute a `(N, 3)` vector array
- decide what you want for glyph scaling (magnitude? abs error?)

### Add controls without breaking rebuild stability

Guidelines:
- Use `state.setdefault(...)` for defaults.
- Add your state keys to the `@state.change(...)` decorator.
- Keep rebuild code exception-safe; a thrown exception will break updates and can cause a blank view.

### Add “shared” ranges between left and right

Right now each view computes its own `clim`.

If you want honest comparisons:
- use a single range for both views for the *same scalar mode*
- e.g., compute `clim_shared = (min(left_scalars,right_scalars), max(...))`

## Debugging tips

- If the view is blank:
  - keep the wireframe sphere actor enabled (it’s a sanity check)
  - print `state.last_error` (the UI already shows it)

- If colors look wrong:
  - ensure the mapper is coloring by the correct array:
    - `SelectColorArray("scalars")`
  - ensure glyph scaling uses a different scalar array (`"scale"`)

- If startup is flaky with `conda run`:
  - run using the environment’s python executable directly.

## Quick glossary

- **LUT**: Lookup table / transfer function mapping scalars → colors
- **Glyph**: A small mesh (arrow/sphere) copied to many points
- **Renderer**: Camera + actor collection
- **Viewport**: A sub-rectangle of the render window
- **VTK.js**: JavaScript/WebGL rendering of VTK scenes in the browser
