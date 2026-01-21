import numpy as np
import pyvista as pv
import seaborn as sns
from quantumresponsepro import analysis
import pandas as pd
from pathlib import Path
import math
from scipy.integrate import lebedev_rule

# New composable plotting + metrics API
from . import beta_plotting




def create_molecule_plot(
    plotter, symbols, coords, center=(0, 0, 0), element_colors=None, atom_radius=0.2
):
    """
    Create a PyVista Plotter with the molecule drawn (colored by element).
    Returns:
        pv.Plotter: Plotter instance with the molecule.
    """
    p = plotter
    # Default colors
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

    positions = coords + np.array(center).reshape(1, 3)

    # Plot each element type
    for elem in unique_elements:
        mask = [i for i, s in enumerate(symbols) if s == elem]
        pts = positions[mask]
        pd = pv.PolyData(pts)
        glyph = pd.glyph(scale=False, orient=False, geom=pv.Sphere(radius=atom_radius))
        color = element_colors.get(elem, "gray")
        p.add_mesh(
            glyph,
            color=color,
        )

    return p


def get_vector_data(mol, basis, freq, vec_shg):
    """
    Get the vector data for a given molecule, basis set, and frequency.
    """
    return vec_shg[mol][basis][freq]


def build_basis_data(mol, omegas, basis_list, vec_shg, offset=(0.0, 0, 0)):
    """
    Assemble basis_vectors & basis_origins for a given molecule and set of omegas.

    Parameters
    ----------
    mol : hashable
        Your molecule key to pass into get_vector_data().
    omegas : list of float
        Frequencies to include.
    basis_list : list of str
        Names of basis‐sets to include (e.g. ['aug-cc-pVDZ','d-aug-cc-pVDZ']).
        'mra-high' is always included automatically.
    offsety : float, optional
        Vertical offset for placing each arrow field.

    Returns
    -------
    basis_vectors : dict
        { 'basis_name(omega)': np.array([...]), … }
    basis_origins : dict
        { 'basis_name(omega)': np.array([x, offsety, 0]), … }
    """
    basis_vectors = {}
    basis_origins = {}

    # always include the MRA reference
    full_basis_list = ["mra-high"] + basis_list
    (x, y, z) = offset

    for omega in omegas:
        # x‐position is evenly spaced between 0 and 1
        offsetx = float(omega) / float(len(omegas))
        for basis in full_basis_list:
            key = f"{basis}({omega})"
            # grab the vector
            basis_vectors[key] = get_vector_data(mol, basis, omega, vec_shg)
            # set its origin
            basis_origins[key] = np.array([x, y, z])

    return basis_vectors, basis_origins


def beta_proj(beta, E):
    # Backwards-compatible re-export
    return beta_plotting.beta_proj(beta, E)


def beta_df_to_np(beta_df):
    # Backwards-compatible re-export
    return beta_plotting.beta_df_to_np(beta_df)


def plot_beta_normals_on_sphere(
    plotter,
    mol,
    basis,
    omega,
    shg_pivot: pd.DataFrame,
    err_pivot: pd.DataFrame = pd.DataFrame(),
    is_error=False,
    center=(0, 0, 0),
    radius=1.5,
    vector_scale=0.3,
    n_phi=60,
    n_theta=60,
    beta_cmap="viridis",
    error_cmap="coolwarm",
    cb_args=None,
    vmax=None,
):
    """
    Plots projected β (or error) vectors on a sphere, colored by their magnitudes.

    Parameters
    ----------
    plotter : pyvista.Plotter
        The PyVista plotter instance.
    mol, basis, omega : hashable
        Keys to look up in the data pivots.
    shg_pivot : pandas.DataFrame
        MultiIndex DataFrame with 27 β components per (mol, basis, omega).
    err_pivot : pandas.DataFrame, optional
        Same shape as shg_pivot, but containing error tensor components.
        Required if is_error=True.
    beta_proj : callable
        Function (tensor, direction) -> projected 3-vector.
    is_error : bool
        If False, plots reference β; if True, plots error vectors.
    center : tuple of float
        Sphere center.
    radius : float
        Sphere radius.
    vector_scale : float
        Scale factor for arrow lengths.
    mra_norm : float
        Normalization factor for reference magnitudes.
    n_phi, n_theta : int
        Resolution of the sphere mesh.
    beta_cmap : str
        Colormap for reference magnitudes.
    error_cmap : str
        Colormap for error magnitudes.
    cb_args : dict, optional
        scalar_bar_args passed to add_mesh (position/size).
    """

    # 1. Select pivot and tensor series
    pivot = err_pivot if is_error else shg_pivot
    tensor = beta_plotting.tensor_from_pivot(pivot, mol, basis, omega)

    # 2. Build sphere and translate
    sphere = pv.Sphere(radius=radius, theta_resolution=n_theta, phi_resolution=n_phi)
    if True:
        z0_sphere = pv.Sphere(
            radius=radius,
            center=(0, 0, 0),
            theta_resolution=3 * n_phi,
            phi_resolution=3,
        )
        sphere = sphere.merge(z0_sphere, tolerance=0.1)

    # 3. Project tensor onto each direction
    directions = sphere.points
    projected, _ = beta_plotting.project_tensor_on_points(tensor, directions)
    magnitudes = np.linalg.norm(projected, axis=1)

    # 4. Scale vectors and attach to mesh
    vectors = np.zeros_like(projected)
    vectors = projected * vector_scale

    sphere["beta_vectors"] = vectors
    sphere.set_active_vectors("beta_vectors")
    if vmax is None:
        vmax = np.max(magnitudes) * 1.1

    # 5. Attach and name scalar
    if is_error:
        scalar_name = f"error_magnitude"
        cmap = error_cmap
    else:
        scalar_name = f"beta_magnitude"
        cmap = beta_cmap
    sphere[scalar_name] = magnitudes

    # 6. Glyph arrows and plot
    bar_args = (cb_args or {}).copy()
    bar_args["title"] = scalar_name.replace("_", " ").title()
    arrow_src = pv.Arrow(
        # # tip_length=0.3,    # fraction of total arrow length
        # tip_radius=0.05,   # relative radius
        # shaft_radius=0.010, # base radius
    )
    sphere.points = sphere.points * 4  # translate points by vectors
    sphere.translate(center, inplace=True)

    # 2) glyph it: orient by your vectors, scale by the magnitude array
    glyphs = sphere.glyph(
        orient="beta_vectors",
        scale=scalar_name,  # uses your magnitudes
        factor=vector_scale,  # base length scaling
        geom=arrow_src,
    )

    plotter.add_mesh(
        glyphs,
        lighting=False,
        scalars=scalar_name,
        clim=[0, vmax],  # set color limits
        cmap=cmap,
        # specular=0.5,
        show_scalar_bar=True,
        opacity=0.6,
    )
    # after you call plot_beta_normals_on_sphere for each basis…
    center_text = center + np.array(
        [0, 0, radius * 1.2]
    )  # raise it a bit above the sphere
    plotter.add_point_labels(
        points=np.array([center_text]),  # your sphere’s center, e.g. (0, 2.5, 0)
        labels=[basis],  # a list of strings, here just one
        font_size=10,
        point_color=None,  # don’t draw the little point marker
        point_size=0,
        text_color="white",
        name=f"label_{basis}",
    )

    # 7. Add subtle sphere for context
    plotter.add_mesh(
        sphere, color="grey", opacity=0.1, lighting=False, show_scalar_bar=False
    )


def signed_scale_errors(
    errors,
    e_min=0.005,  # minimum magnitude (percent)
    e_max=10.0,  # maximum magnitude (percent)
    v_min=1.0,  # mapped magnitude min
    v_max=10.0,  # mapped magnitude max
):
    """
    Log‐map signed error values so that:
      -  |error| in [e_min, e_max] → magnitude in [v_min, v_max]
      -  sign(error) is preserved
      -  error=0 maps to 0

    Parameters
    ----------
    errors : array_like
        Signed error values (e.g. percent errors, can be negative).
    e_min, e_max : float
        Bounds on the magnitude of errors. Anything smaller than e_min is
        clamped up to e_min; anything larger than e_max is clamped down to e_max.
    v_min, v_max : float
        Target range for the mapped magnitudes. Must satisfy 0 < v_min < v_max.

    Returns
    -------
    np.ndarray
        An array of the same shape as `errors`, with values in
        [-v_max, -v_min] ∪ {0} ∪ [v_min, v_max].
    """
    errors = np.asarray(errors, dtype=float)
    signs = np.sign(errors)  # −1, 0, or +1
    mags = np.abs(errors)

    # clamp magnitude into [e_min, e_max]
    mags_clamped = np.clip(mags, e_min, e_max)

    # compute log‐normalized t ∈ [0,1]
    log_min = np.log(e_min)
    log_max = np.log(e_max)
    t = (np.log(mags_clamped) - log_min) / (log_max - log_min)

    # scale to [v_min, v_max]
    scaled_mag = v_min + t * (v_max - v_min)

    # # zero stays zero
    # scaled_mag[mags == 0] = 0.0

    return signs * scaled_mag


def rotation_matrix_from_vectors(vec1, vec2):
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (
        vec2 / np.linalg.norm(vec2)
    ).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    if s == 0:
        return np.eye(3)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + kmat + kmat @ kmat * ((1 - c) / (s**2))


def compare_basis_set_errors_dynamic_grid(
    mol,
    omega,
    shg_pivot,
    mols,
    reference_basis="mra",
    other_bases=["aug-cc-pVDZ", "aug-cc-pVTZ"],
    epsilon=0.05,
    error_epsilon=0.05,
    mine=0.01,  # minimum error value
    maxe=1,  # maximum error value
    max_cols=3,  # max number of columns in grid
    basis_cmap="coolwarm",
):
    # Backwards-compatible wrapper around the new modular implementation.
    # `epsilon` maps to the glyph scaling factor used by the plot builder.
    metric_kwargs = {"mine": mine, "maxe": maxe}
    return beta_plotting.plot_basis_comparison_grid(
        mol=mol,
        omega=omega,
        shg_pivot=shg_pivot,
        mols=mols,
        reference_basis=reference_basis,
        other_bases=list(other_bases),
        quad_order=59,
        ref_scale_order=125,
        glyph_factor=epsilon,
        metric="signed_log_parallel_error",
        metric_kwargs=metric_kwargs,
        metric_cmap=basis_cmap,
        max_cols=max_cols,
        background="white",
    )
