"""Pydantic v2 models for MADNESS input file sections.

.. warning:: This file is **auto-generated** by ``scripts/gen_param_models.py``.
   Do not edit the generated sections by hand — re-run the codegen script instead.
   Hand-written additions between ``# --- BEGIN HAND-WRITTEN ---`` and
   ``# --- END HAND-WRITTEN ---`` markers are preserved across regeneration.

Generated: 2026-03-20 11:52:15 UTC
Source headers:
  DFT:      /gpfs/projects/rjh/adrian/development/madness-worktrees/molresponse-feature-next/src/madness/chem/CalculationParameters.h
  Response: /gpfs/projects/rjh/adrian/development/madness-worktrees/molresponse-feature-next/src/madness/chem/ResponseParameters.hpp
  Molecule: /gpfs/projects/rjh/adrian/development/madness-worktrees/molresponse-feature-next/src/madness/chem/molecule.h
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class DFTSection(BaseModel):
    """Parameters for the ``dft`` section of a MADNESS .in file."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prefix: str = Field(default='mad', description="prefixes your output/restart/json/plot/etc files")

    charge: float = Field(default=0.0, description="total molecular charge")

    xc: str = Field(default='hf', description="XC input line")

    hfexalg: Literal['multiworld', 'multiworld_row', 'fetch_compute', 'smallmem', 'largemem'] = Field(default='multiworld_row', description="hf exchange algorithm")

    memory: list[str] = Field(default=['storefunction', 'nodereplicated', 'distributed'], description="memory algorithm for storing functions (storing,cloud,target)")

    smear: float = Field(default=0.0, description="smearing parameter")

    econv: float = Field(default=1e-05, description="energy convergence")

    dconv: float = Field(default=0.0001, description="density convergence")

    convergence_criteria: list[str] = Field(default=['bsh_residual', 'total_energy'], description="possible values are: bsh_residual, total_energy, each_energy, density")

    k: int = Field(default=-1, description="polynomial order")

    l: float = Field(default=20.0, description="user coordinates box size")

    deriv: Literal['abgv', 'bspline', 'ble'] = Field(default='abgv', description="derivative method")

    dft_deriv: Literal['abgv', 'bspline', 'ble'] = Field(default='abgv', description="derivative method for gga potentials")

    maxrotn: float = Field(default=0.25, description="step restriction used in autoshift algorithm")

    nvalpha: int = Field(default=0, description="number of alpha virtuals to compute")

    nvbeta: int = Field(default=0, description="number of beta virtuals to compute")

    nopen: int = Field(default=0, description="number of unpaired electrons = nalpha-nbeta")

    maxiter: int = Field(default=25, description="maximum number of iterations")

    nio: int = Field(default=1, description="no. of io servers to use")

    spin_restricted: bool = Field(default=True, description="true if spin restricted")

    plotdens: bool = Field(default=False, description="If true print the density at convergence")

    plotcoul: bool = Field(default=False, description="If true plot the total coulomb potential at convergence")

    localize: Literal['pm', 'boys', 'new', 'canon'] = Field(default='new', description="localization method")

    pointgroup: Literal['c1', 'c2', 'ci', 'cs', 'c2v', 'c2h', 'd2', 'd2h'] = Field(default='c1', description="use point (sub) group symmetry if not localized")

    restart: bool = Field(default=False, description="if true restart from orbitals on disk")

    restartao: bool = Field(default=False, description="if true restart from orbitals projected into AO basis (STO3G) on disk")

    no_compute: bool = Field(default=False, description="if true use orbitals on disk, set value to computed")

    save: bool = Field(default=True, description="if true save orbitals to disk")

    maxsub: int = Field(default=10, description="size of iterative subspace ... set to 0 or 1 to disable")

    orbitalshift: float = Field(default=0.0, description="scf orbital shift: shift the occ orbitals to lower energies")

    npt_plot: int = Field(default=101, description="no. of points to use in each dim for plots")

    plot_cell: str = Field(default='Tensor<double>()', description="lo hi in each dimension for plotting (default is all space)")

    plot_cell: list[float] = Field(default=[], description="lo hi in each dimension for plotting (default is all space)")

    aobasis: str = Field(default='6-31g', description="AO basis used for initial guess (6-31gss, 6-31g, 3-21g, sto-6g, sto-3g)")

    derivatives: bool = Field(default=False, description="if true calculate nuclear derivatives")

    dipole: bool = Field(default=False, description="if true calculate dipole moment")

    conv_only_dens: bool = Field(default=False, description="if true remove bsh_residual from convergence criteria (deprecated)")

    psp_calc: bool = Field(default=False, description="pseudopotential calculation for all atoms")

    pcm_data: str = Field(default='none', description="do a PCM (solvent) calculation")

    ac_data: str = Field(default='none', description="do a calculation with asymptotic correction (see ACParameters class in chem/AC.h for details)")

    pure_ae: bool = Field(default=True, description="pure all electron calculation with no pseudo-atoms")

    print_level: int = Field(default=3, description="0: no output; 1: final energy; 2: iterations; 3: timings; 10: debug")

    molecular_structure: str = Field(default='inputfile', description="where to read the molecule from: inputfile or name from the library")

    nalpha: int = Field(default=-1, description="number of alpha spin electrons")

    nbeta: int = Field(default=-1, description="number of beta  spin electrons")

    nmo_alpha: int = Field(default=-1, description="number of alpha spin molecular orbitals")

    nmo_beta: int = Field(default=-1, description="number of beta spin molecular orbitals")

    lo: float = Field(default=1e-10, description="smallest length scale we need to resolve")

    protocol: list[float] = Field(default=[0.0001, 1e-06], description="calculation protocol")

    gopt: bool = Field(default=False, description="geometry optimizer")

    gtol: float = Field(default=0.0001, description="geometry tolerance")

    gtest: bool = Field(default=False, description="geometry tolerance")

    gval: float = Field(default=1e-05, description="value precision")

    gprec: float = Field(default=0.0001, description="gradient precision")

    gmaxiter: int = Field(default=20, description="optimization maxiter")

    ginitial_hessian: bool = Field(default=False, description="compute inital hessian for optimization")

    algopt: Literal['bfgs', 'cg'] = Field(default='bfgs', description="algorithm used for optimization")

    nv_factor: int = Field(default=1, description="factor to multiply number of virtual orbitals with when automatically decreasing nvirt")

    vnucextra: int = Field(default=2, description="load balance parameter for nuclear pot")

    loadbalparts: int = Field(default=2, description="??")

    nwfile: str = Field(default='none', description="Base name of nwchem output files (.out and .movecs extensions) to read from")


class ResponseSection(BaseModel):
    """Parameters for the ``response`` section of a MADNESS .in file."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    prefix: str = Field(default='response', description="prefixes your output/restart/json/plot/etc files")

    fock_json_file: str = Field(default='moldft.fock.json', description="data file for fock matrix")

    archive: str = Field(default='../moldft.restartdata', description="file to read ground parameters from")

    nwchem: bool = Field(default=False, description="Using nwchem files for intelligent starting guess")

    nwchem_dir: str = Field(default='none', description="Root name of nwchem files for intelligent starting guess")

    print_level: int = Field(default=3, description="0: no output; 1: final energy; 2: iterations; 3: timings; 10: debug")

    kain: bool = Field(default=False, description="Turn on Krylov Accelarated Inexact Newton Solver")

    maxrotn: float = Field(default=0.5, description="Max orbital rotation per iteration")

    maxbsh: float = Field(default=10.0, description="Max bsh residual")

    maxsub: int = Field(default=8, description="size of iterative subspace ... set to 0 or 1 to disable")

    xc: str = Field(default='hf', description="XC input line")

    hfexalg: str = Field(default='multiworld_row', description="hf exchange algorithm: choose from multiworld (default), multiworld_row, smallmem, largemem")

    dconv: float = Field(default=1e-06, description="density convergence")

    step_restrict: bool = Field(default=True, description="Toggles step restriction")

    requested_properties: list[str] = Field(default=['polarizability'], description="properties to calculate (polarizability,hessian, hyperpolarizability, Raman.)")

    beta_shg: bool = Field(default=True, alias="beta.shg", description="compute only SHG beta triplets (omegaB=omegaC, omegaA=-(omegaB+omegaC))")

    beta_or: bool = Field(default=False, alias="beta.or", description="compute only optical-rectification beta triplets (omegaB=0, omegaA=-omegaC)")

    beta_all_triplets: bool = Field(default=False, alias="beta.all_triplets", description="compute full beta triplet grid over all (omegaB, omegaC) pairs")

    state_parallel: Literal['off', 'auto', 'on'] = Field(default='off', description="state-level subgroup scheduling mode (off, auto, on)")

    state_parallel_groups: int = Field(default=1, description="number of processor groups for state-level subgroup scheduling")

    state_parallel_min_states: int = Field(default=4, description="minimum number of generated states before auto state-parallel mode activates")

    state_parallel_property_group: int = Field(default=0, description="subgroup id used for property assembly in state-parallel mode")

    state_parallel_point_start_protocol: int = Field(default=1, description="first protocol index where state-parallel mode may fan out by state-frequency point ownership")

    force_retry_removed_frequencies: bool = Field(default=False, description="allow retry of frequencies previously marked remove_from_frequency_set")

    excited_enable: bool = Field(default=False, alias="excited.enable", description="enable excited-state bundle planning metadata scaffolding")

    excited_num_states: int = Field(default=1, alias="excited.num_states", description="number of excited states to target when enabled")

    excited_tda: bool = Field(default=False, alias="excited.tda", description="use Tamm-Dancoff approximation in excited-state stage")

    excited_guess_max_iter: int = Field(default=5, alias="excited.guess_max_iter", description="maximum iterations for excited-state guess stage")

    excited_maxiter: int = Field(default=20, alias="excited.maxiter", description="maximum iterations for excited-state solve stage")

    excited_maxsub: int = Field(default=8, alias="excited.maxsub", description="subspace size for excited-state iterative solves")

    excited_owner_group: int = Field(default=0, alias="excited.owner_group", description="subgroup lane reserved for excited-state bundle execution")

    property: bool = Field(default=False, description="Compute properties")

    dipole: bool = Field(default=False, description="Compute linear dipole response")

    dipole_frequencies: list[float] = Field(default=[0.0], alias="dipole.frequencies", description="frequencies for dipole response")

    dipole_directions: str = Field(default='xyz', alias="dipole.directions", description="directions for dipole response")

    nuclear: bool = Field(default=False, description="Compute nuclear response")

    nuclear_directions: str = Field(default='xyz', alias="nuclear.directions", description="directions for nuclear response")

    nuclear_frequencies: list[float] = Field(default=[0.0], alias="nuclear.frequencies", description="frequencies for nuclear response")

    quadratic: bool = Field(default=False, description="Compute quadratic response properties from defined perturbations")

    localize: Literal['pm', 'boys', 'new', 'canon'] = Field(default='canon', description="localization method")

    maxiter: int = Field(default=25, description="maximum number of iterations")

    protocol: list[float] = Field(default=[0.0001, 1e-06], description="calculation protocol")

    deriv: Literal['abgv', 'bspline', 'ble'] = Field(default='abgv', description="derivative method")

    dft_deriv: Literal['abgv', 'bspline', 'ble'] = Field(default='abgv', description="derivative method for gga potentials")

    k: int = Field(default=-1, description="polynomial order")

    save: bool = Field(default=False, description="save response orbitals to disk")

    restart: bool = Field(default=False, description="restart from saved response orbitals")


class MoleculeSection(BaseModel):
    """Parameters for the ``molecule`` section of a MADNESS .in file."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source: list[str] = Field(default=['inputfile'], description="where       //            to get the coordinates from: ({inputfile}, {library,xxx},       //            {xyz,xxx.xyz})")

    source_type: Literal['inputfile', 'xyz', 'library'] = Field(default='inputfile', description="where to get the coordinates from")

    source_name: str = Field(default='TBD', description="name of the geometry from the library or the input file")

    eprec: float = Field(default=0.0001, description="smoothing for the nuclear potential")

    units: Literal['atomic', 'angstrom', 'bohr', 'au'] = Field(default='atomic', description="coordinate units")

    field: list[float] = Field(default=[0.0, 0.0, 0.0], description="external electric field")

    no_orient: bool = Field(default=False, description="if true the coordinates will not be reoriented and/or symmetrized")

    symtol: float = Field(default=-0.01, description="distance threshold")

    core_type: Literal['none', 'mcp'] = Field(default='none', description="core potential type")

    psp_calc: bool = Field(default=False, description="pseudopotential calculation for all atoms")

    pure_ae: bool = Field(default=True, description="pure all electron calculation with no pseudo-atoms")


# --- BEGIN HAND-WRITTEN ---
# Everything between BEGIN/END markers is preserved across codegen runs.


class Atom(BaseModel):
    """Single atom entry in the molecule section."""

    symbol: str
    x: float
    y: float
    z: float


# Extend MoleculeSection with atoms list (not from C++ headers)
MoleculeSection.model_rebuild()


class MadnessInputFile(BaseModel):
    """Complete MADNESS .in file represented as a Pydantic model.

    Sections default to empty (all MADNESS defaults).  Use ``from_file``
    to parse an existing file, or construct directly and call ``to_file``.
    """

    dft: DFTSection = Field(default_factory=DFTSection)
    molecule: MoleculeSection = Field(default_factory=MoleculeSection)
    response: ResponseSection = Field(default_factory=ResponseSection)
    atoms: list[Atom] = Field(default_factory=list, description="Atom coordinates for the molecule section")

    @classmethod
    def from_file(cls, path: str | Path) -> "MadnessInputFile":
        """Parse a MADNESS .in file into this model."""
        from gecko.workflow.input_parser import parse_madness_input_file

        return parse_madness_input_file(Path(path))

    def to_file(self, path: str | Path) -> None:
        """Serialize this model to a MADNESS .in file."""
        from gecko.workflow.input_serializer import serialize_madness_input

        Path(path).write_text(serialize_madness_input(self))

    def to_madness_str(self) -> str:
        """Return the MADNESS .in file content as a string."""
        from gecko.workflow.input_serializer import serialize_madness_input

        return serialize_madness_input(self)

    def get(self, dotted_key: str) -> Any:
        """Get a parameter value by dotted key, e.g. ``dft.xc`` or ``response.dipole.frequencies``.

        The first segment is the section name (dft/molecule/response).
        The rest is the parameter key within that section.
        """
        section_name, _, param_key = dotted_key.partition(".")
        if not param_key:
            raise ValueError(f"Key must be section.param, got {dotted_key!r}")

        section = self._get_section(section_name)

        # Try field name first, then alias
        if hasattr(section, param_key.replace(".", "_")):
            return getattr(section, param_key.replace(".", "_"))

        # Search by alias
        for fname, finfo in type(section).model_fields.items():
            alias = finfo.alias or fname
            if alias == param_key:
                return getattr(section, fname)

        raise KeyError(f"Unknown parameter {param_key!r} in section {section_name!r}")

    def set(self, dotted_key: str, value: Any) -> None:
        """Set a parameter value by dotted key.

        The value is coerced to the field's type using the Pydantic model.
        """
        section_name, _, param_key = dotted_key.partition(".")
        if not param_key:
            raise ValueError(f"Key must be section.param, got {dotted_key!r}")

        section = self._get_section(section_name)
        field_name = self._resolve_field_name(section, param_key)

        finfo = type(section).model_fields[field_name]
        coerced = _coerce_value(value, finfo.annotation)
        setattr(section, field_name, coerced)

    def _get_section(self, name: str) -> BaseModel:
        if name == "dft":
            return self.dft
        if name == "response":
            return self.response
        if name == "molecule":
            return self.molecule
        raise KeyError(f"Unknown section {name!r}. Valid: dft, molecule, response")

    @staticmethod
    def _resolve_field_name(section: BaseModel, param_key: str) -> str:
        """Resolve a MADNESS parameter key to its Pydantic field name."""
        py_name = param_key.replace(".", "_")
        if py_name in type(section).model_fields:
            return py_name
        # Search by alias
        for fname, finfo in type(section).model_fields.items():
            alias = finfo.alias or fname
            if alias == param_key:
                return fname
        raise KeyError(f"Unknown parameter {param_key!r}")


def _coerce_value(value: Any, annotation: Any) -> Any:
    """Coerce a string value to the target type annotation."""
    if not isinstance(value, str):
        return value

    import typing

    origin = getattr(annotation, "__origin__", None)
    args = getattr(annotation, "__args__", ())

    # Handle typing.Literal
    if origin is Literal:
        if value in args:
            return value
        raise ValueError(f"{value!r} not in allowed values {args}")

    # Handle list types
    if origin is list:
        elem_type = args[0] if args else str
        # Parse [x,y,z] syntax
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            inner = stripped[1:-1]
        else:
            inner = stripped
        if not inner.strip():
            return []
        items = [x.strip() for x in inner.split(",")]
        return [_coerce_scalar(x, elem_type) for x in items]

    # Handle Optional / Union types (Literal is already caught above)
    if origin is typing.Union:
        for arg in args:
            if arg is type(None):
                continue
            try:
                return _coerce_value(value, arg)
            except (ValueError, TypeError):
                continue
        return value

    # Scalar types
    if annotation is bool or annotation == bool:
        return value.lower() in ("true", "1", "yes")
    if annotation is int or annotation == int:
        return int(value)
    if annotation is float or annotation == float:
        return float(value)
    return value


def _coerce_scalar(value: str, target_type: type) -> Any:
    if target_type is float:
        return float(value)
    if target_type is int:
        return int(value)
    if target_type is bool:
        return value.lower() in ("true", "1", "yes")
    return value


# --- END HAND-WRITTEN ---
