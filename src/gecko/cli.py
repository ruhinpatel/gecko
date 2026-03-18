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

    print(f"\n  codes      : {codes}")
    print(f"  basis_sets : {basis_sets}")
    print(f"  out_dir    : {out_dir.resolve()}")
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

    # Print equivalent calc init command for easy re-run
    cmd_parts = [
        "gecko calc init",
        f"--molecule {molecule}",
        f"--property {property_}",
        f"--code {' '.join(codes)}",
        f"--basis {' '.join(basis_sets)}",
        f"--frequencies {' '.join(str(f) for f in frequencies)}",
        f"--xc {xc}",
        f"--out {out_dir}",
    ]
    if geom_file:
        cmd_parts.append(f"--geom-file {geom_file}")
    if want_slurm == "yes":
        cmd_parts += [
            "--slurm",
            f"--partition {partition}",
            f"--nodes {nodes}",
            f"--tasks-per-node {tasks_per_node}",
            f"--walltime {walltime}",
            f"--madqc-exec {madqc_exec}",
        ]
        if account:
            cmd_parts.append(f"--account {account}")
    print("\nEquivalent command:")
    print("  " + " \\\n    ".join(cmd_parts))

    return 0


# ---------------------------------------------------------------------------
# gecko calc submit
# ---------------------------------------------------------------------------


def _calc_submit_command(args: argparse.Namespace) -> int:
    from gecko.workflow.hpc import submit_job
    from gecko.workflow.jobstore import JobRecord, load_store
    from gecko.workflow.remote import RemoteHost

    calc_root = Path(args.calc_dir)
    store = load_store(calc_root)

    # Collect all .sh scripts under calc_root
    scripts = sorted(calc_root.rglob("run_*.sh"))
    if not scripts:
        print(f"No run_*.sh scripts found under {calc_root}")
        return 1

    host: RemoteHost | None = None
    if args.host:
        host = RemoteHost(
            hostname=args.host,
            username=args.user,
            port=args.ssh_port,
            key_file=args.key_file,
            remote_base_dir=args.remote_dir,
        )

    submitted = 0
    for script in scripts:
        # Infer code and mol_name from directory structure
        parts = script.parts
        code = "madness" if "madness" in parts else "dalton"
        mol_name = _guess_mol_name(script)

        try:
            handle = submit_job(script, host=host)
            record = JobRecord(
                job_id=handle.job_id,
                mol_name=mol_name,
                code=code,
                script_path=str(script),
                remote_dir=handle.remote_dir,
                hostname=handle.hostname,
            )
            store.add(record)
            location = f"@{handle.hostname}" if handle.is_remote else "local"
            print(f"  Submitted [{handle.job_id}] {script.name} ({location})")
            submitted += 1
        except RuntimeError as exc:
            print(f"  ERROR: {script.name}: {exc}")

    print(f"\n{submitted} job(s) submitted.  Tracked in {store.path}")
    return 0


# ---------------------------------------------------------------------------
# gecko calc status
# ---------------------------------------------------------------------------


def _calc_status_command(args: argparse.Namespace) -> int:
    from gecko.workflow.hpc import poll_job, JobHandle
    from gecko.workflow.jobstore import load_store
    from gecko.workflow.remote import RemoteHost

    calc_root = Path(args.calc_dir)
    store = load_store(calc_root)
    records = store.all()

    if not records:
        print(f"No jobs tracked in {store.path}")
        return 0

    # Build host lookup keyed by hostname (for connection reuse)
    _ssh_cache: dict[str, object] = {}

    target = store.active() if not args.all else records

    for record in target:
        if record.is_remote if hasattr(record, "is_remote") else bool(record.hostname):
            host = RemoteHost(
                hostname=record.hostname,
                username=args.user,
                port=args.ssh_port,
                key_file=args.key_file,
            )
            handle = JobHandle(
                job_id=record.job_id,
                hostname=record.hostname,
                remote_dir=record.remote_dir,
                script_path=record.script_path,
            )
        else:
            handle = JobHandle(job_id=record.job_id, script_path=record.script_path)

        try:
            status = poll_job(handle)
        except Exception as exc:
            status = f"error: {exc}"

        store.update(record.job_id, status)
        _status_icon = {"done": "✓", "running": "►", "queued": "…", "failed": "✗"}.get(status, "?")
        location = f"@{record.hostname}" if record.hostname else "local"
        print(
            f"  [{record.job_id:>8}] {_status_icon} {status:<10}  "
            f"{record.mol_name}  {record.code}  {location}"
        )

    print(f"\nStore: {store.path}")
    return 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _guess_mol_name(script_path: Path) -> str:
    """Infer molecule name from script or directory path."""
    # script is typically  <root>/<mol>/<code>/run_<mol>.sh
    # walk up until we find a part that doesn't look like a code/basis name
    skip = {"madness", "dalton"}
    for part in reversed(script_path.parts[:-1]):
        if part not in skip and not part.startswith("aug-") and not part.startswith("d-aug"):
            return part
    return script_path.stem


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

    # Shared SSH args used by submit + status
    def _add_ssh_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--host", default="", help="Remote HPC login node hostname (leave blank for local)")
        p.add_argument("--user", default="", help="SSH username")
        p.add_argument("--ssh-port", type=int, default=22)
        p.add_argument("--key-file", default="", help="Path to SSH private key")
        p.add_argument("--remote-dir", default="~/gecko_calcs",
                       help="Base directory on remote host for uploads")

    # gecko calc submit
    submit_parser = calc_subparsers.add_parser(
        "submit",
        help="Submit generated SLURM scripts (local or remote via SSH)",
    )
    submit_parser.add_argument("calc_dir", help="Calculation root directory containing run_*.sh scripts")
    _add_ssh_args(submit_parser)
    submit_parser.set_defaults(func=_calc_submit_command)

    # gecko calc status
    status_parser = calc_subparsers.add_parser(
        "status",
        help="Poll status of tracked jobs",
    )
    status_parser.add_argument("calc_dir", help="Calculation root directory (contains jobs.json)")
    status_parser.add_argument("--all", action="store_true", default=False,
                               help="Show all jobs including completed ones")
    _add_ssh_args(status_parser)
    status_parser.set_defaults(func=_calc_status_command)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
