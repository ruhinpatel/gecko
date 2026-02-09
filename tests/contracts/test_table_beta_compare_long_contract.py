from __future__ import annotations

from pathlib import Path
import logging

import numpy as np

from gecko.core.load import load_calc
from gecko.tables.builder import TableBuilder

logger = logging.getLogger(__name__)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "load_calc"


def test_table_beta_compare_long_contract() -> None:
    calc_a = load_calc(FIXTURES / "01_mra-d04_n2")
    calc_b = load_calc(FIXTURES / "02_aug-cc-pVDZ_n2")

    tb = TableBuilder([calc_a, calc_b])
    df = tb.compare_beta_long(ref_basis="mra-d04", keys=["mol_id", "omegaA", "omegaB", "omegaC", "ijk"],small_thresh=1e-3)
    assert not df.empty

    required_cols = {"mol_id", "omegaA", "omegaB", "omegaC", "ijk", "ref_basis", "basis", "value", "ref_value", "delta", "rel"}
    assert set(df.columns) >= required_cols

    vals = df["value"].astype(float)
    assert np.isfinite(vals).all()

    view_cols = ["mol_id", "omegaA", "omegaB", "omegaC", "ijk", "ref_basis", "basis", "value", "ref_value", "delta", "rel"]
    df_view = df[view_cols].sort_values(["omegaA", "omegaB", "omegaC", "ijk", "basis"]).reset_index(drop=True)
    logger.info("Beta comparison long table rows=%d", len(df_view))
    logger.info("Beta comparison long table:\n%s", df_view.to_string(index=False))


def test_table_beta_compare_long_multibasis_beta_data() -> None:
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

        beta_df = tb.build_beta()
        if beta_df.empty:
            continue

        basis_vals = beta_df["basis"].dropna().astype(str).tolist()
        has_ref_basis = any(b == ref_basis for b in basis_vals)
        assert has_ref_basis, (
            f"Expected beta rows for ref basis '{ref_basis}' in {mol_dir.name}, "
            f"available bases: {sorted(set(basis_vals))}"
        )

        df = tb.compare_beta_long(ref_basis=ref_basis, keys=["mol_id", "omegaA", "omegaB", "omegaC", "ijk"])

        assert not df.empty
        assert (df["ref_basis"] == ref_basis).all()

        view_cols = ["mol_id", "omegaA", "omegaB", "omegaC", "ijk", "ref_basis", "basis", "value", "ref_value", "delta", "rel"]
        df_view = df[view_cols].sort_values(["omegaA", "omegaB", "omegaC", "ijk", "basis"]).reset_index(drop=True).dropna(subset=["value", "ref_value"])
        logger.info("Beta comparison long table (beta_data/%s) rows=%d", mol_dir.name, len(df_view))
        logger.info("Beta comparison long table (beta_data/%s):\n%s", mol_dir.name, df_view.to_string(index=False))
