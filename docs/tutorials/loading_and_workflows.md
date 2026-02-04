# Loading Calculations + Sketching Workflows

This tutorial is meant to **sketch the intended Gecko API** (and how you might use it) before we finalize any workflow abstractions.

Gecko’s current focus is:
- **Fast, read-only loading** of finished calculations (Dalton directories with one-or-more `.out` files; MADNESS JSON runs).
- **Standardized `Calculation` objects** you can quickly translate into Pandas tables for analysis.
- A simple path toward **multi-step workflows** (e.g. Dalton optimize → Dalton raman) without committing to a heavyweight framework.

## 1) Load a single calculation (directory or file)

Gecko’s main entrypoint is `gecko.core.load.load_calc`.

```python
from gecko.core.load import load_calc

# Load a run directory (MADNESS or Dalton will be auto-detected)
calc = load_calc("path/to/run_dir")

print(calc.code)          # "madness" or "dalton"
print(calc.root)          # resolved Path to the calc dir
print(calc.molecule)      # qcelemental.models.Molecule (if found)
print(calc.meta.keys())   # lightweight metadata (basis, label, warnings, ...)
print(calc.data.keys())   # parsed properties + raw payloads
```

You can also point `load_calc` at a **single artifact file**:

```python
from gecko.core.load import load_calc

# Dalton: load and prefer a specific output file inside its directory
opt = load_calc("raman_paper/H2O/aug-cc-pVDZ/optimize_H2O_aug-cc-pVDZ.out")

# MADNESS: load legacy output.json or modern calc_info.json-shaped files
mad = load_calc("some_run/output.json")
```

## 2) Bulk-load a directory tree

To discover calculation directories without assuming a particular layout, use `iter_calc_dirs`.

```python
from gecko.core.iterators import iter_calc_dirs
from gecko.core.load import load_calc

db_root = "raman_paper"

calcs = []
for run_dir in iter_calc_dirs(db_root):
    try:
        calcs.append(load_calc(run_dir))
    except Exception as exc:
        # In real scripts, capture + report failures, but keep going.
        print(f"Failed: {run_dir} ({type(exc).__name__}: {exc})")

print(len(calcs))
```

## 3) Convert parsed results into tables (Pandas DataFrames)

The `TableBuilder` takes a list of `Calculation` objects and emits long-form tables suitable for analysis/ML.

```python
from gecko.tables.builder import TableBuilder

tb = TableBuilder(calcs)

df_geom = tb.build_geometries()
df_e = tb.build_energy()
df_alpha = tb.build_alpha()
df_beta = tb.build_beta()

print(df_geom.columns)
print(df_beta.head())
```

If you want a “single place to join”, `geom_id`/`mol_id`/`calc_id` are intended as the durable keys (see `gecko.ids`).

## 4) Visualize SHG data (standard viewer)

Gecko includes a property-focused visualization module in `gecko.viz`. The key idea is:
- Build (or load) a standardized long-form SHG table (`shg_ijk.csv`)
- View it in a consistent way via the beta viewer app

### 4.1 Build `shg_ijk.csv` from a directory tree

```python
from pathlib import Path
from gecko.viz.io import build_shg_df_from_db

db_root = Path("calc_nlo_beta")
df = build_shg_df_from_db(db_root, include_geometry=True)

df.to_csv("shg_ijk.csv", index=False)
```

### 4.2 Run the viewer app against a CSV

From the repo root (or any environment where `gecko` is installed):

```bash
python -m gecko.viz.apps.beta_viewer --shg-csv shg_ijk.csv
```

### 4.3 (Optional) Write a reusable “bundle”

Bundles are just a directory containing:
- a normalized `shg_ijk.csv`
- a `geometries.json` map (if geometry info is available)

```python
from pathlib import Path
from gecko.viz.io import write_beta_viewer_bundle

bundle_dir = Path("./beta_bundle")
write_beta_viewer_bundle(df, bundle_dir)
```

Then run:

```bash
python -m gecko.viz.apps.beta_viewer --bundle-dir ./beta_bundle
```

## 5) Dalton “multi-part” calculations: optimize → raman

Gecko does **not** yet define a first-class “workflow” object, but you can already treat “workflow steps” as:
- a directory + a set of artifacts,
- where each artifact can be parsed into a `Calculation`,
- and downstream steps can pick the correct molecule/geometry by selecting the right artifact.

### 5.1 Parse the *optimized* geometry from Dalton output

Dalton parsing currently sets `calc.molecule` from the output (when possible). For optimization outputs, the parser prefers:
- a `.mol` block embedded in the output (if present),
- otherwise the **last** `Molecular geometry (au)` block in the file.

```python
from gecko.core.load import load_calc

opt = load_calc("raman_paper/H2O/aug-cc-pVDZ/optimize_H2O_aug-cc-pVDZ.out")
opt_mol = opt.molecule

print(opt.meta.get("label"), opt.meta.get("basis"))
print(opt.meta.get("molecule_id"))  # stable geometry-based id
```

### 5.2 Load the “raman step” separately

If your raman calculation produces a separate `.out` in the same directory, you can load it directly:

```python
from gecko.core.load import load_calc

ram = load_calc("raman_paper/H2O/aug-cc-pVDZ/raman_H2O_aug-cc-pVDZ.out")
```

At this stage, a practical (and simple) workflow convention is:
- **Optimize step defines the geometry**.
- **Raman step consumes that geometry**, so if a raman run exists you expect it to reference the optimized structure (or at least share `molecule_id`).

## 6) A minimal “planner” sketch (no committed API)

Before adding any new core abstractions, you can prototype the workflow logic as a tiny, pure function:

```python
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class NextAction:
    kind: str           # "run_optimize" | "run_raman" | ...
    directory: Path
    note: str = ""

def next_actions_for_raman_dir(step_dir: str | Path) -> list[NextAction]:
    d = Path(step_dir)
    has_opt = any(p.name.startswith("optimize") and p.suffix == ".out" for p in d.glob("*.out"))
    has_raman = any(p.name.startswith("raman") and p.suffix == ".out" for p in d.glob("*.out"))

    if not has_opt:
        return [NextAction("run_optimize", d, "No optimize output found")]
    if not has_raman:
        return [NextAction("run_raman", d, "Optimize exists; raman missing")]
    return []
```

This keeps the “workflow” layer:
- **read-only and fast** (filesystem checks + parsers),
- **separate from submission** (Slurm / local / cloud),
- and easy to evolve into something like `gecko.workflow.plan(db_root, recipe="raman")` later.

## 7) Contracts (how we keep the API clean during refactors)

The goal is to make each “roadmap” section above a **contract-backed test case**:
- A contract states the minimal inputs + outputs and what is stable (keys/columns/types).
- A test asserts the contract using tiny fixtures (so we can safely delete everything not needed).

If a future change breaks a contract, it should be a deliberate choice: update the contract + update the tests together.

## Suggested next steps (small + focused)

1) Add a dedicated extractor for **Dalton optimized geometry** (explicitly label it as `output_molecule` for optimize runs).
2) Add a small `gecko.workflow` module with *only*:
   - step detection helpers (no submission),
   - “what exists / what’s missing” reporting.
3) Add table extractors for workflow-ish metadata (e.g. `step="optimize"|"raman"`, `parent_calc_id`, `consumes_geom_id`).

Once those are in place, the “submit next job” script can live outside the core package and just consume the planner output.
