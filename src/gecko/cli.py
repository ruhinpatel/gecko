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

    codes: list[str] = ["madness", "dalton"] if args.code == ["both"] else args.code
    basis_sets: list[str] = args.basis or ["aug-cc-pVDZ"]
    freqs = [float(f) for f in (args.frequencies or ["0.0"])]
    out_dir = Path(args.out)

    if args.tier and args.tier != "none":
        from gecko.workflow.params import _load_tier
        dft_params, mol_params = _load_tier(args.tier)
        resp_params = None
    else:
        dft_params, mol_params, resp_params = _load_madness_params(
            getattr(args, "madness_params", None)
        )

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
        tier=getattr(args, "tier", None),
        dft_params=dft_params,
        molecule_params=mol_params,
        response_params=resp_params,
    )

    if args.cluster:
        from gecko.workflow.hpc import load_slurm_profile
        profile_tier = args.tier if args.tier and args.tier != "none" else "medium"
        print(f"Loading SLURM profile for cluster={args.cluster!r}, tier={profile_tier!r} …")
        slurm_cfg = load_slurm_profile(args.cluster, args.molecule, profile_tier)
    else:
        slurm_cfg = SlurmConfig(
            partition=args.partition,
            account=args.account,
            nodes=args.nodes,
            tasks_per_node=args.tasks_per_node,
            walltime=args.walltime,
            madqc_executable=args.madqc_exec,
        )

    if args.cluster or args.slurm:
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
    codes_str = _prompt("Codes (madness, dalton, or both)", default="both")
    _codes_raw = codes_str.strip().lower()
    if _codes_raw == "both":
        codes = ["madness", "dalton"]
    else:
        codes = [c.strip() for c in _codes_raw.split(",") if c.strip()]
    _valid = {"madness", "dalton"}
    _invalid = [c for c in codes if c not in _valid]
    if _invalid:
        print(f"  Unknown code(s): {_invalid!r} — valid choices are: madness, dalton, both")
        return 1

    basis_sets_str = _prompt(
        "Basis sets for DALTON (comma-separated, ignored for MADNESS)",
        default="aug-cc-pVDZ",
    )
    basis_sets = [b.strip() for b in basis_sets_str.split(",")]

    freqs_str = _prompt("Frequencies in Hartree (comma-separated)", default="0.0")
    frequencies = [float(f.strip()) for f in freqs_str.split(",")]

    xc = _prompt("XC functional (hf, b3lyp, pbe0, …)", default="hf")
    out_dir = Path(_prompt("Output directory", default=f"./calcs/{molecule}"))
    params_file = _prompt(
        "MADNESS params file (.toml/.json) — leave blank for defaults", default=""
    )
    dft_params, mol_params, resp_params = _load_madness_params(params_file or None)

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
        dft_params=dft_params,
        molecule_params=mol_params,
        response_params=resp_params,
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
# gecko calc results
# ---------------------------------------------------------------------------


def _calc_results_command(args: argparse.Namespace) -> int:
    import gecko
    from gecko.tables.builder import TableBuilder

    calc_dir = Path(args.calc_dir)
    calc = gecko.load_calc(str(calc_dir))
    builder = TableBuilder([calc])

    property_map = {
        "alpha": builder.build_alpha,
        "beta": builder.build_beta,
        "raman": builder.build_raman,
        "energy": builder.build_energy,
    }

    build_fn = property_map[args.property]
    try:
        df = build_fn()
    except Exception as exc:
        print(f"ERROR: Could not extract {args.property}: {exc}")
        return 1

    if df.empty:
        print(f"No {args.property} data found in {calc_dir}")
        return 0

    if args.format == "csv":
        if args.out:
            out_path = Path(args.out)
            df.to_csv(out_path, index=False)
            print(f"Written to {out_path}")
        else:
            print(df.to_csv(index=False), end="")
    else:
        print(df.to_string(index=False))

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


def _load_madness_params(
    params_file: str | None,
) -> tuple[object, object, object]:
    """Load DFTParams, MoleculeParams, ResponseParams from a TOML/JSON file.

    Returns a 3-tuple ``(dft_params, mol_params, resp_params)``, each of
    which is either a params object or None if not specified in the file.
    """
    if not params_file:
        return None, None, None

    from gecko.workflow.params import DFTParams, MoleculeParams, ResponseParams

    path = Path(params_file)
    raw: dict = {}
    if path.suffix in (".toml",):
        import tomllib
        with open(path, "rb") as fh:
            raw = tomllib.load(fh)
    else:
        import json
        raw = json.loads(path.read_text())

    dft = DFTParams(**raw["dft"]) if "dft" in raw else None
    mol = MoleculeParams(**raw["molecule"]) if "molecule" in raw else None
    resp = ResponseParams(**raw["response"]) if "response" in raw else None
    return dft, mol, resp


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


# ---------------------------------------------------------------------------
# gecko input  (show / get / set / validate / create / diff)
# ---------------------------------------------------------------------------


def _input_show_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile

    inp = MadnessInputFile.from_file(args.file)
    section_filter = getattr(args, "section", None)

    if args.format == "json":
        import json

        if section_filter:
            section = inp._get_section(section_filter)
            data = section.model_dump(by_alias=True)
            if section_filter == "molecule":
                data["atoms"] = [a.model_dump() for a in inp.atoms]
        else:
            data = inp.model_dump(by_alias=True)
        print(json.dumps(data, indent=2))
    else:
        if section_filter:
            # Print just one section in MADNESS format
            from gecko.workflow.input_serializer import _serialize_section, _serialize_atoms

            section = inp._get_section(section_filter)
            lines = _serialize_section(section, type(section).model_fields)
            print(section_filter)
            for line in lines:
                print(f"    {line}")
            if section_filter == "molecule":
                for line in _serialize_atoms(inp.atoms):
                    print(f"    {line}")
            print("end")
        else:
            print(inp.to_madness_str(), end="")

    return 0


def _input_get_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile
    from gecko.workflow.params import _render_value

    inp = MadnessInputFile.from_file(args.file)
    value = inp.get(args.key)
    print(_render_value(value))
    return 0


def _input_set_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile

    inp = MadnessInputFile.from_file(args.file)
    inp.set(args.key, args.value)

    if args.dry_run:
        print(inp.to_madness_str(), end="")
    else:
        out_path = Path(args.output) if args.output else Path(args.file)
        inp.to_file(out_path)
        print(f"Written to {out_path}")

    return 0


def _input_validate_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile

    try:
        inp = MadnessInputFile.from_file(args.file)
        n_atoms = len(inp.atoms)
        n_dft = sum(1 for f in inp.dft.model_fields if getattr(inp.dft, f) != getattr(type(inp.dft)(), f))
        n_resp = sum(1 for f in inp.response.model_fields if getattr(inp.response, f) != getattr(type(inp.response)(), f))
        n_mol = sum(1 for f in inp.molecule.model_fields if getattr(inp.molecule, f) != getattr(type(inp.molecule)(), f))
        print(f"Valid: {args.file}")
        print(f"  dft: {n_dft} non-default params")
        print(f"  response: {n_resp} non-default params")
        print(f"  molecule: {n_mol} non-default params, {n_atoms} atoms")
        return 0
    except Exception as exc:
        print(f"Invalid: {args.file}")
        print(f"  Error: {exc}")
        return 1


def _input_create_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile, Atom

    # Start from existing file or defaults
    if args.from_file:
        inp = MadnessInputFile.from_file(args.from_file)
    else:
        inp = MadnessInputFile()

    # Load geometry from file or molecule library
    if args.geom_file:
        from gecko.workflow.geometry import load_geometry_from_file

        mol = load_geometry_from_file(Path(args.geom_file))
        inp.molecule.units = "angstrom"
        inp.atoms = [
            Atom(symbol=sym, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            for sym, xyz in zip(mol.symbols, mol.geometry * 0.529177249)  # Bohr → Å
        ]
    elif args.molecule:
        from gecko.workflow.geometry import load_geometry_from_file

        # Try molecule library first
        mol_lib = Path("/gpfs/projects/rjh/adrian/development/madness-worktrees/molecules")
        candidates = list(mol_lib.rglob(f"{args.molecule}.mol"))
        if candidates:
            mol = load_geometry_from_file(candidates[0])
            inp.molecule.units = "angstrom"
            inp.atoms = [
                Atom(symbol=sym, x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
                for sym, xyz in zip(mol.symbols, mol.geometry * 0.529177249)
            ]
            print(f"Loaded geometry from {candidates[0]}")
        else:
            print(f"Molecule {args.molecule!r} not found in library. Use --geom-file instead.")
            return 1

    # Apply --set overrides
    for kv in (args.set or []):
        key, _, value = kv.partition("=")
        if not value:
            print(f"Invalid --set format: {kv!r} (expected KEY=VALUE)")
            return 1
        inp.set(key.strip(), value.strip())

    out_path = Path(args.output)
    inp.to_file(out_path)
    print(f"Created {out_path}")
    return 0


def _input_diff_command(args: argparse.Namespace) -> int:
    from gecko.workflow.input_model import MadnessInputFile
    from gecko.workflow.params import _render_value

    inp1 = MadnessInputFile.from_file(args.file1)
    inp2 = MadnessInputFile.from_file(args.file2)

    diffs: list[str] = []
    for section_name in ("dft", "response", "molecule"):
        s1 = inp1._get_section(section_name)
        s2 = inp2._get_section(section_name)
        for field_name, field_info in s1.model_fields.items():
            v1 = getattr(s1, field_name)
            v2 = getattr(s2, field_name)
            if v1 != v2:
                key = field_info.alias or field_name
                diffs.append(
                    f"  {section_name}.{key}: {_render_value(v1)} → {_render_value(v2)}"
                )

    # Compare atoms
    if len(inp1.atoms) != len(inp2.atoms):
        diffs.append(f"  atoms: {len(inp1.atoms)} → {len(inp2.atoms)}")
    else:
        for i, (a1, a2) in enumerate(zip(inp1.atoms, inp2.atoms)):
            if a1 != a2:
                diffs.append(f"  atom[{i}]: {a1.symbol} ({a1.x},{a1.y},{a1.z}) → {a2.symbol} ({a2.x},{a2.y},{a2.z})")

    if diffs:
        print(f"Differences between {args.file1} and {args.file2}:")
        print("\n".join(diffs))
    else:
        print("Files are semantically identical.")

    return 0


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

    # --- input subcommand ---
    input_parser = subparsers.add_parser("input", help="Read, write, and edit MADNESS .in files")
    input_subparsers = input_parser.add_subparsers(dest="input_command", required=True)

    # gecko input show
    show_parser = input_subparsers.add_parser("show", help="Display a parsed MADNESS input file")
    show_parser.add_argument("file", help="Path to .in file")
    show_parser.add_argument("--section", "-s", choices=["dft", "response", "molecule"],
                             help="Show only this section")
    show_parser.add_argument("--format", "-f", choices=["madness", "json"], default="madness",
                             help="Output format (default: madness)")
    show_parser.set_defaults(func=_input_show_command)

    # gecko input get
    get_parser = input_subparsers.add_parser("get", help="Get a single parameter value")
    get_parser.add_argument("file", help="Path to .in file")
    get_parser.add_argument("key", help="Dotted key (e.g. dft.xc, response.dipole.frequencies)")
    get_parser.set_defaults(func=_input_get_command)

    # gecko input set
    set_parser = input_subparsers.add_parser("set", help="Set a parameter and write back")
    set_parser.add_argument("file", help="Path to .in file")
    set_parser.add_argument("key", help="Dotted key (e.g. dft.xc)")
    set_parser.add_argument("value", help="New value")
    set_parser.add_argument("--dry-run", action="store_true", help="Print result without writing")
    set_parser.add_argument("--output", "-o", default="", help="Write to a different file")
    set_parser.set_defaults(func=_input_set_command)

    # gecko input validate
    validate_parser = input_subparsers.add_parser("validate", help="Validate a MADNESS input file")
    validate_parser.add_argument("file", help="Path to .in file")
    validate_parser.set_defaults(func=_input_validate_command)

    # gecko input create
    create_parser = input_subparsers.add_parser("create", help="Create a new MADNESS input file")
    create_parser.add_argument("--output", "-o", required=True, help="Output .in file path")
    create_parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                               help="Set parameters (e.g. dft.xc=b3lyp)")
    create_parser.add_argument("--from-file", default="", help="Start from an existing .in file")
    create_parser.add_argument("--molecule", "-m", default="", help="Molecule name (search library)")
    create_parser.add_argument("--geom-file", default="", help="Path to .xyz/.mol geometry file")
    create_parser.set_defaults(func=_input_create_command)

    # gecko input diff
    diff_parser = input_subparsers.add_parser("diff", help="Semantic diff between two input files")
    diff_parser.add_argument("file1", help="First .in file")
    diff_parser.add_argument("file2", help="Second .in file")
    diff_parser.set_defaults(func=_input_diff_command)

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
    init_parser.add_argument("--code", "-c", choices=["madness", "dalton", "both"], nargs="+", default=["madness"])
    init_parser.add_argument("--basis", "-b", nargs="+", default=["aug-cc-pVDZ"],
                             help="Basis sets for DALTON (ignored for MADNESS)")
    init_parser.add_argument("--frequencies", nargs="+", default=["0.0"],
                             help="Optical frequencies in Hartree")
    init_parser.add_argument("--xc", default="hf", help="XC functional (hf, b3lyp, …)")
    init_parser.add_argument("--out", "-o", required=True, help="Output directory")
    init_parser.add_argument("--geom-file", default="", help="Local .xyz/.mol geometry file (skips PubChem)")
    tier_group = init_parser.add_mutually_exclusive_group()
    tier_group.add_argument(
        "--tier", default="none", choices=["low", "medium", "high", "none"],
        help="Numerical accuracy tier from numerical_settings.json (default: none)",
    )
    tier_group.add_argument(
        "--madness-params", default="",
        help="TOML or JSON file with [dft], [molecule], [response] tables for MADNESS overrides",
    )
    init_parser.add_argument(
        "--cluster", default="",
        choices=["xeonmax", "40core", "96core"],
        help="Load SLURM settings from slurm_profiles.json for this cluster (implies --slurm)",
    )
    init_parser.add_argument("--slurm", action="store_true", default=False,
                             help="Also generate SLURM submission scripts (manual settings)")
    # SLURM options (used when --slurm is passed, ignored when --cluster is used)
    init_parser.add_argument("--partition", default="hbm-long-96core")
    init_parser.add_argument("--account", default="")
    init_parser.add_argument("--nodes", type=int, default=4)
    init_parser.add_argument("--tasks-per-node", type=int, default=8)
    init_parser.add_argument("--walltime", default="08:00:00")
    init_parser.add_argument("--madqc-exec", default="madqc", help="Path to madqc executable")
    init_parser.set_defaults(func=_calc_init_command)

    # gecko calc results
    results_parser = calc_subparsers.add_parser(
        "results",
        help="Extract properties from a completed calculation",
    )
    results_parser.add_argument("calc_dir", help="Path to the calculation directory")
    results_parser.add_argument(
        "--property", "-p", choices=["alpha", "beta", "raman", "energy"], default="alpha",
        help="Property to extract (default: alpha)",
    )
    results_parser.add_argument(
        "--format", "-f", choices=["csv", "table"], default="table",
        help="Output format (default: table)",
    )
    results_parser.add_argument(
        "--out", "-o", default="",
        help="Write output to file (csv format only; default: print to stdout)",
    )
    results_parser.set_defaults(func=_calc_results_command)

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

    # --- fixture subcommand ---
    from gecko.fixture_manager import _validate_all, _compare

    fixture_parser = subparsers.add_parser("fixture", help="Developer validation utilities")
    fixture_subparsers = fixture_parser.add_subparsers(dest="fixture_command", required=True)

    # gecko fixture validate-all
    va_parser = fixture_subparsers.add_parser(
        "validate-all",
        help="Compare completed fixture calcs against reference_db.json",
    )
    va_parser.add_argument(
        "--tier", default="medium", choices=["low", "medium", "high"],
        help="Numerical accuracy tier — sets comparison tolerance (default: medium)",
    )
    va_parser.add_argument(
        "--db", default="",
        help="Path to reference_db.json (default: $GECKO_FIXTURES_DIR/reference_db.json)",
    )
    va_parser.set_defaults(func=_validate_all)

    # gecko fixture compare
    cmp_parser = fixture_subparsers.add_parser(
        "compare",
        help="Diff same fixture run under two builds and report regressions",
    )
    cmp_parser.add_argument("--build1", required=True, help="Path to first calc directory")
    cmp_parser.add_argument("--build2", required=True, help="Path to second calc directory")
    cmp_parser.add_argument(
        "--property", "-p", default="alpha", choices=["alpha", "beta", "energy"],
        help="Property to compare (default: alpha)",
    )
    cmp_parser.add_argument(
        "--tier", default="medium", choices=["low", "medium", "high"],
        help="Tier tolerance for regression detection (default: medium)",
    )
    cmp_parser.set_defaults(func=_compare)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
