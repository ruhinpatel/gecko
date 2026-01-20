# Developer Notes

## Canonical molecule ordering

Gecko canonicalizes atom ordering before creating `qcelemental.models.Molecule` so that
molecule hashes are stable across codes (Dalton/MADNESS) when only atom order differs.

Ordering rule: sort by `(atomic_number, x, y, z)` where coordinates are rounded to
`decimals=10` for stable ordering under tiny floating-point noise. The original (unrounded)
coordinates are preserved in the final molecule.

This does **not** address:
- Translation or rotation differences between geometry representations
- Symmetry-related degeneracy issues

If matching must tolerate rigid transforms, implement a geometry key (e.g., distance-matrix
based) and do not rely solely on QCElemental hashes.
