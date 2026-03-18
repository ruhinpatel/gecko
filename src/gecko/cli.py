from __future__ import annotations

import argparse
from pathlib import Path

from gecko.core.iterators import iter_calc_dirs
from gecko.recipes.shg_csv import build_beta_table


def _build_shg_command(args: argparse.Namespace) -> int:
    db_dir = Path(args.db)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    calc_dirs = list(iter_calc_dirs(db_dir))
    shg_df = build_beta_table(
        calc_dirs,
        shg_only=True,
        add_shg_omega=True,
        shg_start_at=args.start_at,
        shg_tol=args.tol,
        app_compat=True,
        verbose=args.verbose,
        fail_fast=args.fail_fast,
    )

    out_path = out_dir / "shg_ijk.csv"
    shg_df.to_csv(out_path, index=False)
    return 0


# ---------------------------------------------------------------------------
# gecko calc init
# ---------------------------------------------------------------------------


def _calc_init_command(args: argparse.Namespace) -> int:
    from gecko.workflow.geometry import fetch_geometry, load_geometry_from_file
    from gecko.workflow.writers import generate_calc_dir
    from gecko.workflow.hpc import SlurmConfig, write_madness_slurm, write_dalton_slurm

    # Resolve geometry
    if args.geom_file:
        print(f"Loading geometry from {args.geom_file} …")
        mol = load_geometry_from_file(Path(args.geom_file))
    else:
        print(f"Fetching geometry for {args.molecule!r} from PubChem …")
        mol = fetch_geometry(args.molecule)
        print(f"  → {mol.get_molecular_formula()} ({len(mol.symbols)} atoms)")

    codes: list[str] = args.code
    basis_sets: list[str] = args.basis or ["aug-cc-pVDZ"]
    freqs = [float(f) for f in (args.frequencies or ["0.0"])]
    out_dir = Path(args.out)

    print(f"Generating input files in {out_dir} …")
    paths = generate_calc_dir(
        molecule=mol,
        mol_name=args.molecule,
        property=args.property,
        codes=codes,
        basis_sets=basis_sets,
        frequencies=freqs,
        xc=args.xc,
        out_dir=out_dir,
    )

    slurm_cfg = SlurmConfig(
        partition=args.partition,
        account=args.account,
        nodes=args.nodes,
        tasks_per_node=args.tasks_per_node,
        walltime=args.walltime,
        madqc_executable=args.madqc_exec,
    )

    if args.slurm:
        print("Generating SLURM scripts …")
        for in_path in paths.get("madness", []):
            s = write_madness_slurm(in_path, slurm_cfg)
            print(f"  {s}")

        # Pair up .dal / .mol files for dalton
        dal_files = [p for p in paths.get("dalton", []) if p.suffix == ".dal"]
        mol_files = [p for p in paths.get("dalton", []) if p.suffix == ".mol"]
        for dal, mol_f in zip(dal_files, mol_files):
            s = write_dalton_slurm(dal, mol_f, slurm_cfg)
            print(f"  {s}")

    # Summary
    print("\nGenerated files:")
    for code, file_list in paths.items():
        for p in file_list:
            print(f"  [{code}]  {p}")

    return 0


# ---------------------------------------------------------------------------
# gecko calc wizard  (interactive)
# ---------------------------------------------------------------------------


def _calc_wizard_command(_args: argparse.Namespace) -> int:
    from gecko.workflow.geometry import fetch_geometry, load_geometry_from_file
    from gecko.workflow.writers import generate_calc_dir
    from gecko.workflow.hpc import SlurmConfig, write_madness_slurm, write_dalton_slurm

    print("=" * 60)
    print("  gecko calc wizard — interactive calculation setup")
    print("=" * 60)

    molecule = _prompt("Molecule name or formula", default="H2O")
    geom_file = _prompt("Local geometry file (.xyz / .mol) — leave blank to fetch from PubChem", default="")

    if geom_file:
        mol = load_geometry_from_file(Path(geom_file))
    else:
        print(f"  Fetching geometry for {molecule!r} from PubChem …")
        mol = fetch_geometry(molecule)
        print(f"  → {mol.get_molecular_formula()} ({len(mol.symbols)} atoms)")

    property_ = _prompt_choices("Property to compute", ["alpha", "beta", "raman"], default="alpha")
    codes_str = _prompt("Codes (comma-separated: madness, dalton, or both)", default="madness,dalton")
    codes = [c.strip().lower() for c in codes_str.split(",")]

    basis_sets_str = _prompt(
        "Basis sets for DALTON (comma-separated, ignored for MADNESS)",
        default="aug-cc-pVDZ",
    )
    basis_sets = [b.strip() for b in basis_sets_str.split(",")]

    freqs_str = _prompt("Frequencies in Hartree (comma-separated)", default="0.0")
    frequencies = [float(f.strip()) for f in freqs_str.split(",")]

    xc = _prompt("XC functional (hf, b3lyp, pbe0, …)", default="hf")
    out_dir = Path(_prompt("Output directory", default=f"./calcs/{molecule}"))

    print(f"\nGenerating input files in {out_dir} …")
    paths = generate_calc_dir(
        molecule=mol,
        mol_name=molecule,
        property=property_,
        codes=codes,
        basis_sets=basis_sets,
        frequencies=frequencies,
        xc=xc,
        out_dir=out_dir,
    )

    want_slurm = _prompt_choices("Generate SLURM scripts?", ["yes", "no"], default="yes")
    if want_slurm == "yes":
        partition = _prompt("SLURM partition", default="hbm-long-96core")
        account = _prompt("SLURM account (leave blank if not needed)", default="")
        nodes = int(_prompt("Number of nodes", default="4"))
        tasks_per_node = int(_prompt("Tasks per node", default="8"))
        walltime = _prompt("Wall-clock time (HH:MM:SS)", default="08:00:00")
        madqc_exec = _prompt("madqc executable path", default="madqc")

        slurm_cfg = SlurmConfig(
            partition=partition,
            account=account,
            nodes=nodes,
            tasks_per_node=tasks_per_node,
            walltime=walltime,
            madqc_executable=madqc_exec,
        )
        for in_path in paths.get("madness", []):
            s = write_madness_slurm(in_path, slurm_cfg)
            print(f"  SLURM: {s}")
        dal_files = [p for p in paths.get("dalton", []) if p.suffix == ".dal"]
        mol_files = [p for p in paths.get("dalton", []) if p.suffix == ".mol"]
        for dal, mol_f in zip(dal_files, mol_files):
            s = write_dalton_slurm(dal, mol_f, slurm_cfg)
            print(f"  SLURM: {s}")

    print("\nAll done!  Generated files:")
    for code, file_list in paths.items():
        for p in file_list:
            print(f"  [{code}]  {p}")

    return 0


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def _prompt(label: str, default: str = "") -> str:
    display = f" [{default}]" if default else ""
    val = input(f"{label}{display}: ").strip()
    return val if val else default


def _prompt_choices(label: str, choices: list[str], default: str) -> str:
    options = "/".join(
        c.upper() if c == default else c for c in choices
    )
    val = input(f"{label} ({options}): ").strip().lower()
    return val if val in choices else default


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gecko")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- shg subcommand (existing) ---
    shg_parser = subparsers.add_parser("shg", help="SHG table utilities")
    shg_subparsers = shg_parser.add_subparsers(dest="shg_command", required=True)

    build_parser = shg_subparsers.add_parser("build", help="Build SHG ijk table")
    build_parser.add_argument("--db", required=True, help="Database root directory")
    build_parser.add_argument("--out", required=True, help="Output directory")
    build_parser.add_argument("--start-at", type=int, default=0)
    build_parser.add_argument("--tol", type=float, default=1e-12)
    build_parser.add_argument("--fail-fast", action="store_true", default=False)
    build_parser.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=True)
    build_parser.set_defaults(func=_build_shg_command)

    # --- calc subcommand (new) ---
    calc_parser = subparsers.add_parser("calc", help="Calculation setup utilities")
    calc_subparsers = calc_parser.add_subparsers(dest="calc_command", required=True)

    # gecko calc init
    init_parser = calc_subparsers.add_parser(
        "init",
        help="Generate input files for MADNESS / DALTON",
    )
    init_parser.add_argument("--molecule", "-m", required=True, help="Molecule name or formula (e.g. SO2)")
    init_parser.add_argument("--property", "-p", choices=["alpha", "beta", "raman"], default="alpha")
    init_parser.add_argument("--code", "-c", choices=["madness", "dalton"], nargs="+", default=["madness"])
    init_parser.add_argument("--basis", "-b", nargs="+", default=["aug-cc-pVDZ"],
                             help="Basis sets for DALTON (ignored for MADNESS)")
    init_parser.add_argument("--frequencies", nargs="+", default=["0.0"],
                             help="Optical frequencies in Hartree")
    init_parser.add_argument("--xc", default="hf", help="XC functional (hf, b3lyp, …)")
    init_parser.add_argument("--out", "-o", required=True, help="Output directory")
    init_parser.add_argument("--geom-file", default="", help="Local .xyz/.mol geometry file (skips PubChem)")
    init_parser.add_argument("--slurm", action="store_true", default=False,
                             help="Also generate SLURM submission scripts")
    # SLURM options (used when --slurm is passed)
    init_parser.add_argument("--partition", default="hbm-long-96core")
    init_parser.add_argument("--account", default="")
    init_parser.add_argument("--nodes", type=int, default=4)
    init_parser.add_argument("--tasks-per-node", type=int, default=8)
    init_parser.add_argument("--walltime", default="08:00:00")
    init_parser.add_argument("--madqc-exec", default="madqc", help="Path to madqc executable")
    init_parser.set_defaults(func=_calc_init_command)

    # gecko calc wizard
    wizard_parser = calc_subparsers.add_parser(
        "wizard",
        help="Interactive guided setup for a new calculation",
    )
    wizard_parser.set_defaults(func=_calc_wizard_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
