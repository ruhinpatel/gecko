# Gecko

gecko is a modular Python toolkit for loading, comparing, and visualizing quantum-chemical response data across electronic structure codes.

## `gecko` src

```graphsql
src/gecko/
  __init__.py                 # exposes load_calc + a couple convenience imports
  core/
    model.py                  # Calculation (minimal), maybe Molecule later
    load.py                   # load_calc + registry
    iterators.py              # directory scanning helpers (read-only mode)
  plugins/
    madness/
      __init__.py
      detect.py               # can_load rules
      parse.py                # thin wrapper around migrated legacy parser
      legacy/                 # (copy of migration/parsers/madness.py + helpers)
    dalton/
      __init__.py
      detect.py
      parse.py
      legacy/                 # (copy of migration/parsers/dalton*.py)
  recipes/
    shg_csv.py                # replaces the notebook workflow gradually
  viz/
    unit_sphere.py            # core plotting / mapping helpers
    metrics.py                # field_error equivalents
    io.py                     # load shg_ijk.csv / dataframe helpers
  apps/
    trame_shg_viewer.py       # migration/viz/application.py
  workflows/
    templates/                # packaged templates
      dalton/
      madness/
    legacy/                   # quarantine: db/ and cli/ initially
```

## External geometry for legacy outputs

Some legacy MADNESS outputs (e.g., `output.json`) do not embed molecular geometry.
You can provide external `.mol` files and gecko will resolve them in the following
priority order:

1) `--mol-map` (explicit label -> file mapping)
2) `--mol-dir` (label-based lookup, `LABEL.mol`)
3) `--mol-file` (single fallback)

Examples:

gecko-viz --calc-root /path/to/calcs --mol-dir /path/to/mols
gecko-viz --calc-root /path/to/calcs --mol-map /path/to/mol_map.json
gecko-viz --calc-root /path/to/calcs --mol-file /path/to/H2O.mol
