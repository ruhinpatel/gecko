# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (uv recommended)
uv venv && uv pip install -e .

# Run tests
pytest
pytest --cov                          # with coverage
pytest tests/contracts/test_table_beta_contract.py  # single test file

# Lint & format
ruff format src tests
ruff check src tests
mypy src

# CLI
gecko shg build --db /path/to/calcs --out /path/to/output

# Visualization apps
python -m gecko.viz.apps.beta_viewer --shg-csv data/csv_data/shg_ijk.csv
python -m gecko.viz.apps.raman_dashboard --db-dir /path/to/calcs
```

## Architecture

Gecko is a plugin-based Python toolkit for loading, comparing, and visualizing quantum-chemical response data from MADNESS and DALTON electronic structure codes.

### Plugin Dispatch

`load_calc(path)` (exported from `src/gecko/__init__.py`) is the main entry point. It auto-detects the calculation type (MADNESS vs DALTON) via `plugins/*/detect.py` and delegates loading to the appropriate `plugins/*/loader.py`. After loading, a finalization pipeline in `core/load.py` handles molecule resolution, artifact attachment, and metadata enrichment.

### Data Model

`Calculation` (in `core/model.py`) is the central container. `Calculation.data` holds raw legacy parser outputs; `.artifacts` holds discovered file paths. The design is migration-first — legacy parsers live in `plugins/madness/legacy/` and `plugins/dalton/legacy/`.

### Tables

`tables/builder.py` contains `TableBuilder`, which uses extractors from `tables/extractors.py` to pull response properties (beta, alpha, Raman, energy, timing) from loaded calculations into pandas DataFrames. `recipes/shg_csv.py` is a high-level workflow that builds beta tables from a directory of calculations.

### Visualization

Interactive web dashboards in `viz/apps/` use [Trame](https://trame.readthedocs.io/) with VTK rendering. `viz/state.py` manages app state; `viz/vtk_scene.py` handles VTK setup.

### Testing

Tests are contract-based in `tests/contracts/`. Fixtures are real calculation directories under `tests/fixtures/` — `01_mra-d04_n2/` for MADNESS and `02_aug-cc-pVDZ_n2/` for DALTON.

## Current Work in Progress

**Branch:** `claude/init-project-setup-PC09o`

### Feature: `gecko calc` — calculation input file generator

New subcommand for generating MADNESS and DALTON input files. Key files:

- `src/gecko/cli.py` — `gecko calc init` and `gecko calc wizard` CLI commands
- `src/gecko/workflow/writers.py` — `MadnessInput`, `DaltonInput`, `generate_calc_dir`
- `src/gecko/workflow/geometry.py` — PubChem fetch + local .xyz/.mol loading
- `src/gecko/workflow/hpc.py` — SLURM script generation
- `src/gecko/plugins/dalton/legacy/dalton_write_inputs.py` — `.mol` file renderer

### Status

- `gecko calc init --code madness` works end-to-end
- `gecko calc wizard` works for MADNESS; DALTON file generation is **not working** on the user's machine (no error, no files produced)
- Diagnostic prints added to wizard (`codes`, `basis_sets`, `out_dir` printed before generation)
- Wizard now prints the equivalent `gecko calc init` command at the end for easy re-runs
- Fixed: removed invalid `_atoms_formatter` import in `dalton_write_inputs.py`

### Next step

User needs to pull branch, run the wizard selecting dalton, and share what the diagnostic lines print:
```
  codes      : [...]
  basis_sets : [...]
  out_dir    : ...
```
That output will reveal why dalton files are not being generated.
