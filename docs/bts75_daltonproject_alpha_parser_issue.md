# BTS-75: DaltonProject OutputParser — Alpha Polarizability Bug

**Author:** Ruhin Patel  
**Date:** 2026-04-13  
**Context:** BTS-75 (DaltonProject Integration) — Step 2 evaluation: replacing Gecko's legacy Dalton output parser with `daltonproject.dalton.OutputParser`  
**Status:** Blocking partial replacement of alpha parser; beta replacement unaffected

---

## Background

As part of BTS-75 we are replacing Gecko's custom Dalton I/O with `daltonproject.dalton`.
Step 1 (input generator) is complete — `DaltonInput` now uses `daltonproject.dalton.program.molecule_input()`
and `dalton_input()` internally instead of hand-written renderers.

For Step 2 (output parser), we evaluated whether `daltonproject.dalton.output_parser.OutputParser`
can parse existing `.out` files produced by Dalton on Seawulf without re-running the calculation.

---

## What We Tested

| Property | Method | Result |
|---|---|---|
| Beta (hyperpolarizability) | `result.first_hyperpolarizability` | **Works** — (3×3×3) tensor matches legacy parser exactly |
| Alpha, static (ω=0 only) | `result.polarizabilities` | **Works** — tensor and isotropic value correct |
| Alpha, multi-frequency, linear molecule (N₂) | `result.polarizabilities` | **Works** — symmetry collapses to one RSPLR block |
| Alpha, multi-frequency, non-linear molecule (H₂O, NH₃O) | `result.polarizabilities` | **Fails** — `IndexError: index 5 is out of bounds for axis 0 with size 5` |

Test files used:
- `tests/fixtures/load_calc/02_aug-cc-pVDZ_n2/quad_n2-aug-cc-pVDZ.out` (beta, works)
- A multi-frequency alpha output for a non-linear molecule (H₂O or NH₃O) — not included in the repo; reproduce on Seawulf with any 5-frequency Dalton run on a C1-symmetry molecule

---

## Root Cause

### How Dalton structures a linear response output

When Dalton computes frequency-dependent polarizability, it solves the linear response equations
separately for each dipole operator (XDIPLEN, YDIPLEN, ZDIPLEN) and each requested frequency.
For a non-linear molecule with no symmetry reduction, this produces **three independent RSPLR blocks**
in the output, one per operator:

```
RSPLR -- operator label : XDIPLEN
RSPLR -- frequencies    :  0.0  0.015  0.030  0.045  0.060    ← block 1

  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.015
  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.030
  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.045
  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.060

RSPLR -- operator label : ZDIPLEN
RSPLR -- frequencies    :  0.0  0.015  0.030  0.045  0.060    ← block 2

  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.015
  ...                                                           ← markers repeat

RSPLR -- operator label : YDIPLEN                             ← block 3
  ...
  @ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY : 0.015
  ...                                                           ← markers repeat

@ -<< XDIPLEN  ; XDIPLEN  >> =  9.9278    ← consolidated final results (all freqs)
@ -<< YDIPLEN  ; YDIPLEN  >> =  9.9278
@ -<< ZDIPLEN  ; ZDIPLEN  >> =  8.4702
...
```

For 5 frequencies and 3 operators, there are **4 × 3 = 12** `@ FREQUENCY DEPENDENT` markers
across all three blocks.

### The bug in `OutputParser.polarizabilities`

The relevant code in `daltonproject/dalton/output_parser.py` (lines 119–145):

```python
frequencies = []
with open(f'{self.filename}.out', 'r') as output_file:
    for line in output_file:
        if 'RSPLR -- frequencies' in line:
            freq_strs = line.split()
            frequencies.extend([float(freq) for freq in freq_strs[4:]])
            break                                      # reads first block only

    polarizability_mat = np.zeros((len(frequencies), 3, 3))   # e.g. shape (5, 3, 3)
    polarizability = np.zeros((3, 3))

    for line in output_file:
        polarizability = polar_tensor(polarizability, line)
        cond1 = '@ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :'
        cond2 = 'Total CPU  time used in RESPONSE'
        if cond1 in line or cond2 in line:
            polarizability_mat[0] = polarizability
            break

    count = 1
    for line in output_file:
        if '@ FREQUENCY DEPENDENT ...' in line:
            count += 1            # ← increments on EVERY marker across ALL blocks
        polarizability = polar_tensor(polarizability, line)
        polarizability_mat[count] = polarizability   # ← IndexError when count >= 5
        if 'Total CPU  time used in RESPONSE' in line:
            break
```

**The assumption:** one RSPLR block → N-1 frequency markers for N frequencies.
**The reality:** three RSPLR blocks → 3×(N-1) markers for non-linear molecules.

With 5 frequencies, `polarizability_mat` has shape `(5, 3, 3)` (valid indices 0–4).
But `count` reaches 13 as it hits all 12 markers, causing `IndexError` at `polarizability_mat[5]`.

### Why it worked on N₂

N₂ is a linear molecule with D∞h symmetry. The X and Y dipole operators are equivalent by symmetry
and Dalton collapses them into a single block. Only one RSPLR block is produced, so the counter
stays in bounds. This masked the bug entirely in Gecko's existing test suite.

---

## Impact on BTS-75

- **Beta replacement:** Unaffected. `result.first_hyperpolarizability` works correctly on all tested files.
- **Alpha replacement:** Blocked for multi-frequency calculations on non-linear molecules
  (which is the majority of real use cases — H₂O, C₆H₆, naphthalene, etc.).

---

## Proposed Fix

The correct approach is to **ignore the intermediate RSPLR blocks entirely** and parse only the
consolidated final results section, which appears once at the end of the output regardless of
molecular symmetry:

```
@ -<< XDIPLEN  ; XDIPLEN  >> =   9.9278D+00
@ -<< XDIPLEN  ; YDIPLEN  >> =   0.0000D+00
...
@ FREQUENCY DEPENDENT SECOND ORDER PROPERTIES WITH FREQUENCY :   0.015022
@ -<< XDIPLEN  ; XDIPLEN  >> =   9.9450D+00
...
```

A corrected `polarizabilities` implementation would:
1. Scan for `RSPLR -- frequencies` once to get the frequency list (unchanged)
2. Skip directly to the final consolidated results block
3. Parse frequency markers and tensor components from that block only

This avoids counting markers across multiple RSPLR blocks entirely.

---

## Alternatives Considered

### Option A — Partial replacement (current plan)
Keep Gecko's legacy alpha parser (`plugins/dalton/parse.py`) as-is.
Replace only beta with `result.first_hyperpolarizability`.

- **Pro:** Zero risk, ships immediately, no dependency on upstream fix
- **Con:** ~400 lines of custom regex for alpha remain in Gecko indefinitely

### Option B — Fix DaltonProject upstream
File a bug report and PR to DaltonProject with the corrected final-results parsing approach.
Once merged and released, remove the legacy alpha parser from Gecko.

- **Pro:** Fixes it at the source for all DaltonProject users
- **Con:** Dependent on upstream review cycle; timeline unknown

### Option C — Patch locally in Gecko
Subclass or monkeypatch `OutputParser` in Gecko with a corrected implementation,
without waiting for upstream.

- **Pro:** Full fix without upstream dependency
- **Con:** Maintenance burden; must be removed once upstream is fixed

---

## Recommendation

**Option A now + Option B in parallel:**

1. Ship beta-only parser replacement as part of BTS-75 Step 2 (unblocks BTS-75 progress)
2. File a bug report to DaltonProject upstream with the suggested final-results fix
3. Once DaltonProject releases a patch, remove the legacy alpha parser in a follow-up PR

This gives the cleanest long-term path without blocking the current sprint.

