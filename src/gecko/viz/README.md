# gecko.viz

Reusable visualization helpers and the Trame-based beta viewer.

## Run with an existing CSV

If running from the repo root, ensure the package is installed in your environment:

```bash
python -m pip install -e .
```

```bash
python -m gecko.viz.apps.beta_viewer --shg-csv data/csv_data/shg_ijk.csv
```

## Build from a database directory

```bash
python -m gecko.viz.apps.beta_viewer --db-dir /path/to/calcs
```

## Build a reusable bundle

```bash
python -m gecko.viz.apps.beta_viewer --db-dir /path/to/calcs --write-bundle ./beta_bundle
python -m gecko.viz.apps.beta_viewer --bundle-dir ./beta_bundle
```

## Notes
- SHG omega indexing uses $\omega_B = \omega_C$ and starts at 0.
- MADNESS basis is labeled as "MRA".
