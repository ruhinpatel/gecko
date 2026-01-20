# SHG Table Pipeline

This document describes the canonical SHG (second-harmonic generation) table pipeline
that replaces the reference workflow in the nlo_test.ipynb notebook.

## Why SHG needs a derived omega index

The SHG subset is defined by rows where $\omega_B = \omega_C$. Visualization requires
an integer index per unique SHG frequency for efficient filtering and plotting. The
index is derived from the unique $\omega_B$ values within each molecule group.

## Per-molecule indexing

The omega index is assigned **per molecule** so that the same frequency index refers
to the same physical molecule. Grouping uses:

1. `geom_id` (QCElemental molecule hash)
2. `mol_id` (QCElemental formula)
3. `root` as a fallback

## Why indexing starts at 0

The notebook workflow enumerates unique $\omega_B$ values starting at 0. The pipeline
preserves this behavior for deterministic parity with existing visualization logic.

## Pipeline overview

The pipeline implemented in `gecko.tables.shg` is:

1. `build_beta_long` – load calculations and emit a long-form beta table
2. `filter_shg_rows` – keep only rows where $|\omega_B - \omega_C| \leq \text{tol}$
3. `assign_shg_omega_index` – map unique $\omega_B$ values to integer `omega`

## How this replaces nlo_test.ipynb

The notebook now serves as a reference implementation. The code in
`gecko.tables.shg` reproduces its logic without relying on directory layout or
interactive steps.

## Visualization consumption

The visualization app expects a `shg_ijk.csv` with:
- raw frequencies (`omegaA`, `omegaB`, `omegaC`)
- components (`ijk`)
- values (`value`)
- metadata (`geom_id`, `mol_id`, `code`, `basis`)
- SHG index (`omega`)

The CLI command:

```
gecko shg build --db <database_dir> --out <out_dir>
```

writes the canonical `shg_ijk.csv` to the output directory.
