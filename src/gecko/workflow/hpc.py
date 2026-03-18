"""SLURM script generation and job submission for HPC clusters."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gecko.workflow.remote import RemoteHost


@dataclass
class JobHandle:
    """Identifies a submitted SLURM job.

    Parameters
    ----------
    job_id : str
        SLURM job ID returned by ``sbatch``.
    hostname : str
        Login-node hostname, or empty string for local submissions.
    remote_dir : str
        Remote working directory, or empty string for local submissions.
    script_path : str
        Local path of the SLURM ``.sh`` script.
    """

    job_id: str
    hostname: str = ""
    remote_dir: str = ""
    script_path: str = ""

    @property
    def is_remote(self) -> bool:
        return bool(self.hostname)


@dataclass
class SlurmConfig:
    """Parameters for a SLURM batch job.

    Parameters
    ----------
    partition : str
        SLURM partition / queue name.
    nodes : int
        Number of nodes to request.
    tasks_per_node : int
        MPI tasks per node.
    walltime : str
        Wall-clock time limit in ``HH:MM:SS`` format.
    account : str
        SLURM account / allocation (optional).
    module_load_script : str
        Shell script sourced to load modules (e.g. ``~/load_xeonmax.sh``).
    mad_num_threads : int
        Value of ``MAD_NUM_THREADS`` environment variable for MADNESS.
    madqc_executable : str
        Path (or name) of the ``madqc`` executable.
    dalton_executable : str
        Path (or name) of the ``dalton`` executable.
    dalton_memory_gb : int
        Memory in GB passed to ``dalton -gb``.
    extra_env : dict[str, str]
        Additional environment variables to export.
    """

    partition: str = "hbm-long-96core"
    nodes: int = 4
    tasks_per_node: int = 8
    walltime: str = "08:00:00"
    account: str = ""
    module_load_script: str = "~/load_xeonmax.sh"
    mad_num_threads: int = 10
    madqc_executable: str = "madqc"
    dalton_executable: str = "dalton"
    dalton_memory_gb: int = 10
    extra_env: dict[str, str] = field(default_factory=dict)


def generate_madness_slurm(input_file: Path, config: SlurmConfig) -> str:
    """Return a SLURM batch script string for a MADNESS job.

    Parameters
    ----------
    input_file : Path
        Path to the ``.in`` file (used for job name and ``--input`` flag).
    config : SlurmConfig
        SLURM and runtime settings.

    Returns
    -------
    str
        Complete SLURM script content.
    """
    input_file = Path(input_file)
    job_name = input_file.stem
    total_tasks = config.nodes * config.tasks_per_node

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks-per-node={config.tasks_per_node}",
        f"#SBATCH --time={config.walltime}",
        f"#SBATCH --output={job_name}.out",
        f"#SBATCH --error={job_name}.err",
        f"#SBATCH --partition={config.partition}",
    ]
    if config.account:
        lines.append(f"#SBATCH --account={config.account}")

    lines += [
        "",
        "module purge",
        f"source {config.module_load_script}",
        "",
        f"export MAD_NUM_THREADS={config.mad_num_threads}",
    ]
    for key, val in config.extra_env.items():
        lines.append(f"export {key}={val}")

    lines += [
        "",
        f'mpirun -np {total_tasks} --map-by numa "{config.madqc_executable}" '
        f"--wf=response --input={input_file}",
    ]

    return "\n".join(lines) + "\n"


def generate_dalton_slurm(
    dal_file: Path,
    mol_file: Path,
    config: SlurmConfig,
) -> str:
    """Return a SLURM batch script string for a DALTON job.

    Parameters
    ----------
    dal_file : Path
        Path to the ``.dal`` input file.
    mol_file : Path
        Path to the ``.mol`` geometry file.
    config : SlurmConfig
        SLURM and runtime settings.

    Returns
    -------
    str
        Complete SLURM script content.
    """
    dal_file = Path(dal_file)
    mol_file = Path(mol_file)
    job_name = f"dalton_{dal_file.stem}"
    total_tasks = config.nodes * config.tasks_per_node

    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --nodes={config.nodes}",
        f"#SBATCH --ntasks-per-node={config.tasks_per_node}",
        f"#SBATCH --time={config.walltime}",
        f"#SBATCH --output={job_name}.out",
        f"#SBATCH --error={job_name}.err",
        f"#SBATCH --partition={config.partition}",
    ]
    if config.account:
        lines.append(f"#SBATCH --account={config.account}")

    lines += [
        "",
        "module purge",
        f"source {config.module_load_script}",
        "",
    ]
    for key, val in config.extra_env.items():
        lines.append(f"export {key}={val}")

    lines += [
        "",
        f'echo "Running DALTON: {dal_file.stem}"',
        f'echo "Input: {dal_file}  Molecule: {mol_file}"',
        f'echo "{"-" * 50}"',
        "",
        f"{config.dalton_executable} -N {total_tasks} -gb {config.dalton_memory_gb} "
        f"-dal {dal_file} -mol {mol_file}",
    ]

    return "\n".join(lines) + "\n"


def write_madness_slurm(input_file: Path, config: SlurmConfig, out_dir: Path | None = None) -> Path:
    """Write MADNESS SLURM script to disk and return its path."""
    input_file = Path(input_file)
    dest = Path(out_dir) if out_dir else input_file.parent
    dest.mkdir(parents=True, exist_ok=True)
    script_path = dest / f"run_{input_file.stem}.sh"
    script_path.write_text(generate_madness_slurm(input_file, config))
    script_path.chmod(0o755)
    return script_path


def write_dalton_slurm(
    dal_file: Path,
    mol_file: Path,
    config: SlurmConfig,
    out_dir: Path | None = None,
) -> Path:
    """Write DALTON SLURM script to disk and return its path."""
    dal_file = Path(dal_file)
    dest = Path(out_dir) if out_dir else dal_file.parent
    dest.mkdir(parents=True, exist_ok=True)
    script_path = dest / f"run_{dal_file.stem}.sh"
    script_path.write_text(generate_dalton_slurm(dal_file, mol_file, config))
    script_path.chmod(0o755)
    return script_path


def submit_job(
    script_path: Path,
    host: RemoteHost | None = None,
) -> JobHandle:
    """Submit a SLURM script via ``sbatch`` and return a :class:`JobHandle`.

    Parameters
    ----------
    script_path : Path
        Local path to the SLURM ``.sh`` script.
    host : RemoteHost or None
        If provided, upload the script's parent directory to the remote host
        and run ``sbatch`` there via SSH.  Requires ``paramiko``.
        If ``None``, run ``sbatch`` locally.

    Returns
    -------
    JobHandle
    """
    script_path = Path(script_path)
    if host is not None:
        from gecko.workflow.remote import submit_remote_job
        job_id = submit_remote_job(script_path, host)
        # remote_dir is <host.remote_base_dir>/<script_parent.name>
        import posixpath
        from gecko.workflow.remote import open_ssh, _expand_remote_tilde, _ssh_run
        ssh = open_ssh(host)
        try:
            base = _expand_remote_tilde(host.remote_base_dir, ssh)
        finally:
            ssh.close()
        remote_dir = posixpath.join(base, script_path.parent.name)
        return JobHandle(
            job_id=job_id,
            hostname=host.hostname,
            remote_dir=remote_dir,
            script_path=str(script_path),
        )

    # Local submission
    result = subprocess.run(
        ["sbatch", str(script_path)],
        capture_output=True,
        text=True,
        cwd=str(script_path.parent),
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"sbatch failed (exit {result.returncode}):\n{result.stderr.strip()}"
        )
    # sbatch outputs: "Submitted batch job 12345"
    parts = result.stdout.strip().split()
    job_id = parts[-1]
    return JobHandle(job_id=job_id, script_path=str(script_path))


def poll_job(handle: JobHandle | str) -> str:
    """Query SLURM job status.

    Parameters
    ----------
    handle : JobHandle or str
        A :class:`JobHandle` (supports remote polling) or a plain job ID
        string (local only).

    Returns one of: ``"queued"``, ``"running"``, ``"done"``, ``"failed"``,
    or ``"unknown"``.
    """
    if isinstance(handle, str):
        return _poll_local(handle)

    if handle.is_remote:
        from gecko.workflow.remote import RemoteHost, poll_remote_job
        host = RemoteHost(hostname=handle.hostname, username="")  # username needed at connect
        return poll_remote_job(handle.job_id, host)

    return _poll_local(handle.job_id)


def _poll_local(job_id: str) -> str:
    result = subprocess.run(
        ["squeue", "--job", job_id, "--format=%T", "--noheader"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return "unknown"
    output = result.stdout.strip().upper()
    if not output:
        return "done"
    if output in ("PENDING", "CF"):
        return "queued"
    if output in ("RUNNING",):
        return "running"
    if output in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
        return "failed"
    return output.lower()
