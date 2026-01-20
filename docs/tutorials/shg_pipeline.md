# SHG Table Pipeline Tutorial

## 1) What Problem This Solves

Second-harmonic generation (SHG) hyperpolarizability data describe how a molecule responds to an applied optical field and generates a signal at twice the frequency. In practice, SHG data arrive from multiple calculations, often across different codes (MADNESS and Dalton), and spread across many directories. To analyze or visualize these results consistently, you need a **long-form table** that standardizes the data layout and preserves raw frequencies.

For SHG, we care about rows where $\omega_B = \omega_C$. A derived integer **omega index** is used so visualization tools can quickly filter, group, and compare SHG responses per molecule across multiple frequencies. This tutorial shows how to build that table in a deterministic, reproducible way.

---

## 2) Data Model Overview

**Core concepts in gecko:**

- **Calculation**: one parsed result (MADNESS or Dalton) with associated metadata.
- **Molecule**: a `qcelemental.models.Molecule` built from parsed geometry.
- **Identity**:
  - `geom_id = molecule.get_hash()`
  - `mol_id = molecule.formula`
- **No user-defined labels**: gecko does not store or require human-readable molecule names.
- **Basis**:
  - MADNESS → `basis = "MRA"` (standardized label)
  - Dalton → basis parsed from output text

Schematic-style view (textual):

```
Calculation (code, basis, root)
  └── Molecule (qcelemental)
         ├── geom_id = get_hash()
         └── mol_id  = formula
```

---

## 3) Input Assumptions

### Option A: Database Directory
A directory containing many calculation subdirectories and/or files.

- MADNESS identified by `output.json` or `*.calc_info.json`
- Dalton identified by `*.out` files
- Geometry is parsed from outputs and converted to `qcelemental.models.Molecule`

Choose this option when you want to **parse and build the table directly** from calculation outputs.

### Option B: Pre-built CSV
A previously generated `shg_ijk.csv`.

Choose this option when you want **fast loading for visualization or analysis** and do not need to re-parse calculations.

---

## 4) End-to-End Tutorial (Concrete Example)

Assume a directory named `calc_nlo_beta/` that contains a mix of MADNESS and Dalton calculations across multiple subdirectories.

### Step 1: Build the SHG Table

```python
from pathlib import Path
import gecko
from gecko.tables.shg import build_shg_ijk

db = Path("calc_nlo_beta")

# Recursively pass all paths; gecko will detect valid calcs
shg_df = build_shg_ijk(db.rglob("*"))

print(shg_df.head())
```

**What `rglob("*")` does:** it walks the directory tree and yields files/directories. The SHG pipeline detects valid MADNESS and Dalton calculations from these paths.

**What rows represent:** each row is a single hyperpolarizability component (`ijk`) at a specific frequency tuple $(\omega_A, \omega_B, \omega_C)$, with metadata attached.

**Key columns:**
- `geom_id`: molecule geometry hash
- `mol_id`: molecule formula
- `code`: `madness` or `dalton`
- `basis`: `MRA` (MADNESS) or parsed basis (Dalton)
- `omegaA`, `omegaB`, `omegaC`: raw frequencies
- `ijk`: tensor component label
- `value`: hyperpolarizability value
- `root`: calculation root path

### Step 2: Inspect Omega Indexing

```python
shg_df[["mol_id", "omegaB", "omega"]].drop_duplicates().sort_values(
    ["mol_id", "omega"]
)
```

**What this shows:**
- Omega indexing is **per molecule**
- The index starts at **0**
- The mapping is **deterministic** for a given dataset

### Step 3: Write CSV Output

```python
out = Path("data/csv_data")
out.mkdir(parents=True, exist_ok=True)

shg_df.to_csv(out / "shg_ijk.csv", index=False)
```

### Step 4: Load from CSV Later

```python
import pandas as pd

shg_df = pd.read_csv("data/csv_data/shg_ijk.csv")
```

This is the recommended workflow for visualization: **no recalculation or parsing required**.

---

## 5) Relationship to Visualization App

The trame / PyVista visualization app consumes `shg_ijk.csv` directly. It expects the following columns:

- `geom_id`
- `mol_id`
- `code`
- `basis`
- `omegaA`, `omegaB`, `omegaC`
- `omega` (derived SHG index)
- `ijk`
- `value`

These are sufficient for:
- SHG frequency filtering
- Basis set comparisons
- Per-molecule analysis
- Component-wise plotting

---

## 6) Design Notes and Future Extensions

### Bitwise geometry identity
`geom_id` is based on `molecule.get_hash()`. This is **bitwise geometry equality** after canonical ordering. It does **not** account for rotation or translation. This may evolve in the future if rigid-transform invariance is required.

### Generalization to other properties
The same long-form approach applies to other properties (alpha, Raman, etc.). The SHG pipeline is one instance of a general pattern: **one table per property** with shared metadata.

### Molecule-centric and property-centric views
Gecko’s design supports both:
- **Molecule-centric**: group by `geom_id` / `mol_id`
- **Property-centric**: filter by `code`, `basis`, `omega`, `ijk`

This makes it easy to extend the pipeline without changing how downstream tools consume the CSV.
