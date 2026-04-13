# daltonproject.dalton — API Notes

Brief reference compiled from the daltonproject docs, source code, and tutorials.
Source: https://daltonproject.readthedocs.io | https://gitlab.com/daltonproject/daltonproject

---

## Installation

```bash
pip install git+https://gitlab.com/daltonproject/daltonproject
```

Requires the Dalton binary on `PATH` and MKL shared libraries on `LD_LIBRARY_PATH`.

---

## Core Workflow

```python
import daltonproject as dp
import numpy as np

result = dp.dalton.compute(molecule, basis, qc_method, properties,
                           compute_settings=None,
                           filename="my_calc",
                           force_recompute=False)
```

Arguments are type-checked — order matters.

---

## Molecule

```python
# From inline geometry string (Angstrom, semicolon-separated)
mol = dp.Molecule(atoms="O 0.0 0.0 0.117; H 0.0 0.757 -0.469; H 0.0 -0.757 -0.469",
                  charge=0, multiplicity=1)

# From file (.xyz, .mol, .gau)
mol = dp.Molecule(input_file="water.xyz", charge=0, multiplicity=1)
```

Key attributes: `mol.elements`, `mol.coordinates`, `mol.charge`, `mol.multiplicity`

---

## Basis

Uses [Basis Set Exchange (BSE)](https://www.basissetexchange.org) under the hood.
Any BSE name is valid (case-insensitive).

```python
basis_dz   = dp.Basis(basis="cc-pVDZ")        # small, fast
basis_daug = dp.Basis(basis="d-aug-cc-pVDZ")  # doubly-augmented, better for polarizability
```

**Note:** `d-aug-cc-pVDZ` is not available in BSE for Li (Z=3). Use `aug-cc-pVDZ` for Li instead.

---

## QCMethod

```python
method = dp.QCMethod("HF")    # Hartree-Fock
method = dp.QCMethod("B3LYP") # DFT
```

### Open-shell (ROHF)

`daltonproject` does not have a `uhf=True` flag. Open-shell is handled via ROHF:
set `multiplicity > 1` on the Molecule and call `scf_occupation()` on the method.

```python
# Li: 1s² 2s¹ — 1 closed shell, 1 open shell
li  = dp.Molecule(atoms="Li 0.0 0.0 0.0", charge=0, multiplicity=2)
method = dp.QCMethod("HF")
method.scf_occupation(closed_shells=[1], open_shells=[1])

# NO: 15 electrons — 7 closed, 1 open
no  = dp.Molecule(atoms="N 0.0 0.0 0.0; O 0.0 0.0 1.151", charge=0, multiplicity=2)
method = dp.QCMethod("HF")
method.scf_occupation(closed_shells=[7], open_shells=[1])

# OH: 9 electrons — 4 closed, 1 open
oh  = dp.Molecule(atoms="O 0.0 0.0 0.0; H 0.0 0.0 0.970", charge=0, multiplicity=2)
method = dp.QCMethod("HF")
method.scf_occupation(closed_shells=[4], open_shells=[1])
```

This writes `.DOUBLY`/`.SINGLY` keywords into the Dalton `*SCF INPUT` block.

---

## Property

```python
# Static polarizability (omega=0)
prop = dp.Property(polarizabilities=True)

# Frequency-dependent (frequencies in Hartree)
prop = dp.Property(polarizabilities={"frequencies": [0.0, 0.0656, 0.0932]})
```

Other available properties: `energy`, `dipole`, `gradients`, `hessian`,
`excitation_energies`, `first_hyperpolarizability`, `nmr_shieldings`, `optical_rotations`

---

## ComputeSettings

```python
settings = dp.ComputeSettings(
    work_dir="/path/to/calc/dir",   # IMPORTANT: must match directory of input files
    scratch_dir="/tmp",             # or $DALTON_TMPDIR / $SCRATCH
    mpi_num_procs=1,
    omp_num_threads=4,
    memory=8000,                    # MB
)
```

**Important:** `work_dir` defaults to `os.getcwd()`. Dalton resolves `-mol`/`-dal` paths
relative to `work_dir`, so always set it to the directory where the input files live
and pass just the filename stem (not a full path) to `filename=`.

The `DALTON_LAUNCHER` env variable overrides how the binary is invoked:
```bash
export DALTON_LAUNCHER="mpirun -np 1"
```

---

## Output Parser

`dp.dalton.compute()` returns a `daltonproject.dalton.OutputParser` instance.

### Polarizability

```python
pol = result.polarizabilities       # NamedTuple(frequencies, values)
# pol.frequencies -> np.ndarray shape (n_freqs,)  in Hartree
# pol.values      -> np.ndarray shape (n_freqs, 3, 3)

alpha = pol.values[0]               # static (omega=0) 3x3 tensor
alpha_xx  = alpha[0, 0]
alpha_yy  = alpha[1, 1]
alpha_zz  = alpha[2, 2]
alpha_iso = np.trace(alpha) / 3.0
```

Internally the parser scans the Dalton `.out` file for lines like:
```
@ -<< XDIPLEN  ; XDIPLEN  >> =  ...
```
and fills the symmetric 3×3 tensor. Fortran `D` exponents are converted to Python floats.

### Other result attributes

```python
result.energy                   # SCF/MP2/CC total energy (Hartree)
result.dipole                   # np.ndarray shape (3,) in Debye
result.excitation_energies      # np.ndarray
result.first_hyperpolarizability # np.ndarray shape (3,3,3)
result.gradients
result.hessian
result.nmr_shieldings
```

---

## Batch Calculations

```python
molecules = [he, h2, lih, h2o, c6h6, naphthalene, li, no, oh]
names     = ["He", "H2", "LiH", "H2O", "C6H6", "naphthalene", "Li", "NO", "OH"]

results = dp.dalton.compute_farm(
    molecules, basis, method, prop,
    filenames=names,
    force_recompute=False,
)

for name, res in zip(names, results):
    print(name, np.trace(res.polarizabilities.values[0]) / 3)
```

Runs one shared `.dal` file with separate `.mol` files, parallelized across available CPUs.

---

## Environment Setup (Seawulf)

```bash
export PATH="/gpfs/projects/rjh/adrian/dalton/build:/cm/shared/apps/openmpi4/gcc/4.1.5/bin:$PATH"
export LD_LIBRARY_PATH="/gpfs/software/intel/oneAPI/2024_2/mkl/2024.2/lib/intel64:$LD_LIBRARY_PATH"
export DALTON_LAUNCHER="mpirun -np 1"
```
