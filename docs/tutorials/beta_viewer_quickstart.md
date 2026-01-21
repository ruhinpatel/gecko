# Beta Viewer Quickstart

This quickstart shows how to build the SHG table and launch the Trame-based beta viewer.

## 1) Build `shg_ijk.csv` using the reuse-first recipe

```python
from pathlib import Path
from gecko.core.iterators import iter_calc_dirs
from gecko.recipes.shg_csv import build_beta_table

root = Path("calc_nlo_beta")
calc_dirs = list(iter_calc_dirs(root))

shg_df = build_beta_table(
    calc_dirs,
    shg_only=True,
    add_shg_omega=True,
    shg_start_at=0,
    shg_tol=1e-12,
    include_geometry=True,
    app_compat=True,
    verbose=True,
)

out = Path("data/csv_data")
out.mkdir(parents=True, exist_ok=True)
shg_df.to_csv(out / "shg_ijk.csv", index=False)
```

## 2) Launch the viewer from CSV

```bash
python -m gecko.viz.apps.beta_viewer --shg-csv data/csv_data/shg_ijk.csv
```

## 3) Launch the viewer from a database directory

```bash
python -m gecko.viz.apps.beta_viewer --db-dir /path/to/calcs
```

## Optional: build a reusable bundle

```bash
python -m gecko.viz.apps.beta_viewer --db-dir /path/to/calcs --write-bundle ./beta_bundle
python -m gecko.viz.apps.beta_viewer --bundle-dir ./beta_bundle
```

## Omega indexing rule

- SHG rows are defined by $|\omega_B - \omega_C| \leq 10^{-12}$.
- The integer `omega` index starts at 0 and is assigned per molecule.
