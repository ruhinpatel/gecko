from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from gecko.core.load import load_calc
from gecko.tables.builder import TableBuilder

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "load_calc"


def test_table_alpha_compare_long_contract() -> None:
    calc_a = load_calc(FIXTURES / "03_mra-raman_h2o")
    calc_b = load_calc(FIXTURES / "05_dalton_raman_h2o")

    tb = TableBuilder([calc_a, calc_b])
    df = tb.compare_alpha_long(ref_basis="mra-p07", keys=["mol_id", "omega", "ij"],small_thresh=1e-3)
    assert not df.empty

    required_cols = {"mol_id", "omega", "ij", "ref_basis", "basis", "value", "ref_value", "delta", "rel"}
    assert set(df.columns) >= required_cols

    for col in ["value", "ref_value", "delta"]:
        vals = df[col].astype(float)
        assert np.isfinite(vals).all()

    view_cols = ["mol_id", "omega", "ij", "ref_basis", "basis", "value", "ref_value", "delta", "rel"]
    df_view = df[view_cols].sort_values(["omega", "ij", "basis"]).reset_index(drop=True)
    logger.info("Alpha comparison long table rows=%d", len(df_view))
    logger.info("Alpha comparison long table:\n%s", df_view.to_string(index=False))


def test_table_alpha_compare_long_multibasis_beta_data() -> None:
    beta_root = FIXTURES.parent / "beta_data"
    mol_dirs = [p for p in sorted(beta_root.iterdir()) if p.is_dir()]
    ref_basis = "mra-d06"

    for mol_dir in mol_dirs:
        calc_dirs = [p for p in sorted(mol_dir.iterdir()) if p.is_dir()]
        calcs = [load_calc(p) for p in calc_dirs]
        tb = TableBuilder(calcs)

        calc_basis = [str(c.meta.get("basis")) for c in calcs if c.meta.get("basis") is not None]
        assert ref_basis in calc_basis, (
            f"Expected loaded calc metadata with basis '{ref_basis}' in {mol_dir.name}, "
            f"loaded bases: {sorted(set(calc_basis))}"
        )

        alpha_df = tb.build_alpha()
        if alpha_df.empty:
            continue

        basis_cols = sorted(set(alpha_df["basis"].dropna().astype(str).tolist()))
        assert ref_basis in basis_cols, (
            f"Expected alpha rows for ref basis '{ref_basis}' in {mol_dir.name}, "
            f"available bases: {basis_cols}"
        )
        # Compare tables require at least one non-reference basis.
        if len(basis_cols) < 2:
            continue

        df = tb.compare_alpha_long(
            ref_basis=ref_basis,
            keys=["mol_id", "omega", "ij"],
            small_thresh=1e-3,
        )

        assert not df.empty
        assert (df["ref_basis"] == ref_basis).all()

        view_cols = ["mol_id", "omega", "ij", "ref_basis", "basis", "value", "ref_value", "delta", "rel"]
        df_view = df[view_cols].sort_values(["omega", "ij", "basis"]).reset_index(drop=True)
        logger.info("Alpha comparison long table (beta_data/%s) rows=%d", mol_dir.name, len(df_view))
        logger.info("Alpha comparison long table (beta_data/%s):\n%s", mol_dir.name, df_view.to_string(index=False))
