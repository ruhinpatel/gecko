from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np
import qcelemental as qcel


def canonicalize_atom_order(
    symbols: Iterable[str],
    geometry: np.ndarray | Iterable[Iterable[float]],
    *,
    decimals: int = 10,
) -> tuple[list[str], np.ndarray]:
    """
    Canonicalize atom ordering for stable hashing/lookup.

    This sorts atoms by (atomic_number, x, y, z) after rounding coordinates
    to the given decimal precision. It does NOT correct for rotations or
    translations; only atom order permutations.
    """
    symbols_list = [str(s) for s in symbols]
    coords = np.asarray(geometry, dtype=float).reshape(-1, 3)

    xyz_r = np.round(coords, decimals=decimals)
    atomic_numbers = np.array([qcel.periodictable.to_Z(sym) for sym in symbols_list], dtype=int)

    order = np.lexsort((xyz_r[:, 2], xyz_r[:, 1], xyz_r[:, 0], atomic_numbers))

    symbols_sorted = [symbols_list[i] for i in order]
    coords_sorted = coords[order]

    return symbols_sorted, coords_sorted
