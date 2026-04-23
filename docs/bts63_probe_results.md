# BTS-63: daltonproject Probe — Results & Notes

**Task:** Validate that `daltonproject.dalton` can reproduce polarizability for all 9 fixture molecules.
**Script:** `gecko/scripts/daltonproject_probe.py`
**Output:** scratch directory on Seawulf (pass via `--scratch`)
**API reference:** `gecko/docs/daltonproject_api_notes.md`

---

## How to Run

```bash
# Set up Seawulf environment (paths will differ per user/cluster setup)
export PATH="<dalton_build_dir>:/cm/shared/apps/openmpi4/gcc/4.1.5/bin:$PATH"
export LD_LIBRARY_PATH="<mkl_lib_dir>:$LD_LIBRARY_PATH"
export DALTON_LAUNCHER="mpirun -np 1"

.venv/bin/python scripts/daltonproject_probe.py \
    --basis both \
    --scratch <scratch_dir>
```

Re-run a subset:
```bash
.venv/bin/python scripts/daltonproject_probe.py --mol H2O LiH --basis dz --force
```

---

## What Was Tested

- **9 molecules:** He, H2, LiH, H2O, C6H6, naphthalene, Li, NO, OH
- **Method:** HF (Hartree-Fock)
- **Property:** Static polarizability (alpha, omega=0)
- **2 basis sets:** `cc-pVDZ` (small/fast) and `d-aug-cc-pVDZ` (doubly-augmented, production)
- **Open-shell:** Li, NO, OH run as ROHF doublets via `method.scf_occupation()`

---

## Results

### cc-pVDZ — all 9 completed

| Molecule | alpha_xx | alpha_yy | alpha_zz | alpha_iso |
|----------|----------|----------|----------|-----------|
| He | 0.3087 | 0.3087 | 0.3087 | 0.3087 |
| H2 | 1.1651 | 1.1651 | 6.4281 | 2.9194 |
| LiH | 24.2142 | 24.2142 | 17.6749 | 22.0344 |
| H2O | 3.0401 | 6.9171 | 5.0917 | 5.0163 |
| C6H6 | 72.2598 | 72.2598 | 24.5319 | 56.3505 |
| naphthalene | 118.6574 | 145.8188 | 37.7831 | 100.7531 |
| Li | 167.9524 | 167.9524 | 167.9524 | 167.9524 |
| NO | 4.5598 | 4.7839 | 12.2891 | 7.2110 |
| OH | 2.6135 | 2.3529 | 5.5494 | 3.5053 |

### d-aug-cc-pVDZ — 7/9 completed

| Molecule | alpha_xx | alpha_yy | alpha_zz | alpha_iso | MRA ref | diff |
|----------|----------|----------|----------|-----------|---------|------|
| He | 1.3288 | 1.3288 | 1.3288 | 1.3288 | — | — |
| H2 | 4.5148 | 4.5148 | 6.5226 | 5.1841 | — | — |
| LiH | — | — | — | **ERROR** | 24.1366 | BSE missing Li |
| H2O | 7.9498 | 9.1495 | 8.4660 | 8.5218 | 8.5417 | 0.02 ✓ |
| C6H6 | 79.8950 | 79.8950 | 45.4927 | 68.4276 | — | — |
| naphthalene | 129.5722 | 158.2018 | 67.3530 | 118.3757 | — | — |
| Li | — | — | — | **ERROR** | — | BSE missing Li |
| NO | 9.1764 | 8.7738 | 14.8065 | 10.9189 | — | — |
| OH | 5.8164 | 6.3505 | 7.7701 | 6.6457 | — | — |

---

## Where to Find Output Files

```
<scratch_dir>/
└── <molecule>/
    └── <basis>/
        ├── <mol>_<basis>.dal   # Dalton input (wave function + response)
        ├── <mol>_<basis>.mol   # Geometry + basis
        ├── <mol>_<basis>.out   # Full Dalton output log
        └── <mol>_<basis>.tar.gz
```

---

## Before vs After daltonproject

| | Before (Gecko custom) | After (daltonproject) |
|---|---|---|
| Input generation | `DaltonInput` class in `gecko/src/gecko/workflow/writers.py` — manually builds `.dal` and `.mol` files | `dp.dalton.compute()` handles input generation internally |
| Output parsing | `gecko/src/gecko/plugins/dalton/legacy/` — custom regex parser for Dalton `.out` files | `OutputParser` built into daltonproject — `result.polarizabilities.values[0]` |
| Open-shell | `dft` block had no spin flag — **Li, NO, OH failed** (BTS-67 blocker) | `method.scf_occupation(closed_shells, open_shells)` — **all 3 work** |
| Basis sets | Hardcoded in `.mol` file template | BSE-backed — any basis name via `dp.Basis("name")` |
| Maintainability | Custom parsers to maintain | Upstream package handles it |

---

## Issues Found

1. **LiH and Li with d-aug-cc-pVDZ fail** — `d-aug-cc-pVDZ` is not available for Li (Z=3) in BSE. Fix: use `aug-cc-pVDZ` for Li-containing systems with this basis family.

2. **`work_dir` bug in daltonproject** — `ComputeSettings.work_dir` defaults to `os.getcwd()`. Dalton prepends it to input file paths, causing a double-path error. Fixed by passing `ComputeSettings(work_dir=calc_dir)` with a relative filename stem. See `scripts/daltonproject_probe.py` lines ~214-224.

3. **Env setup required on Seawulf** — Dalton binary and MKL libs are not on default PATH. Must set the 3 env vars listed in the "How to Run" section above before every run.

---

## Conclusion

`daltonproject` is a viable replacement for Gecko's custom Dalton I/O:
- Works for all 9 fixture molecules (16/18 basis combinations)
- ROHF open-shell support solves the BTS-67 blocker for Li/NO/OH
- H2O with d-aug matches MRA reference to 0.2% — production quality
- The 2 failures are a BSE coverage gap for Li, not a daltonproject limitation
