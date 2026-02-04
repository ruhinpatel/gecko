from __future__ import annotations

from pathlib import Path

import gecko
from gecko.core import iterators
from gecko.recipes.shg_csv import build_beta_table


def test_build_beta_table_shg_omega_mapping():
    root = Path("tests/fixtures/dalton_qr")
    df = build_beta_table(
        root,
        shg_only=True,
        add_shg_omega=True,
        shg_start_at=0,
        shg_tol=1e-12,
        verbose=False,
    )

    assert not df.empty
    assert (df["omegaB"] - df["omegaC"]).abs().max() <= 1e-12

    for mol_id, group in df.groupby("mol_id"):
        if mol_id is None:
            continue
        assert group["omega"].min() == 0


def test_build_beta_table_uses_iterators(monkeypatch):
    root = Path("tests/fixtures/dalton_qr")
    called = {"count": 0}

    original = iterators.iter_calc_dirs

    def wrapped(path):
        called["count"] += 1
        return original(path)

    monkeypatch.setattr(iterators, "iter_calc_dirs", wrapped)

    df = build_beta_table(root, verbose=False)
    assert called["count"] >= 1
    assert not df.empty


def test_dalton_multi_out_counts():
    root = Path("tests/fixtures/dalton_multi_outputs")
    out_files = list(root.glob("*.out"))
    calcs = gecko.load_calcs(root)

    assert len(calcs) == 1
    outputs = calcs[0].data.get("dalton_outputs") or {}
    assert len(outputs) == len(out_files)
