"""SSH-based remote HPC operations.

Requires the optional ``paramiko`` package::

    pip install gecko[hpc]
    # or
    pip install paramiko
"""

from __future__ import annotations

import posixpath
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import paramiko


@dataclass
class RemoteHost:
    """SSH connection parameters for a remote HPC login node.

    Parameters
    ----------
    hostname : str
        Hostname or IP of the login node (e.g. ``"hpc.example.edu"``).
    username : str
        SSH username.
    port : int
        SSH port (default 22).
    key_file : str
        Path to private key file.  If empty, uses the SSH agent or
        ``~/.ssh/id_rsa`` / ``~/.ssh/id_ed25519``.
    remote_base_dir : str
        Base directory on the remote host where calculation directories
        are uploaded (e.g. ``"/scratch/username/gecko_calcs"``).
    """

    hostname: str
    username: str
    port: int = 22
    key_file: str = ""
    remote_base_dir: str = "~/gecko_calcs"


def open_ssh(host: RemoteHost) -> paramiko.SSHClient:
    """Open and return an authenticated SSH connection.

    The caller is responsible for calling ``.close()`` on the returned client.
    """
    _require_paramiko()
    import paramiko  # noqa: PLC0415

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict = dict(
        hostname=host.hostname,
        port=host.port,
        username=host.username,
    )
    if host.key_file:
        connect_kwargs["key_filename"] = host.key_file
    client.connect(**connect_kwargs)
    return client


def upload_directory(local_dir: Path, host: RemoteHost, ssh: paramiko.SSHClient) -> str:
    """Upload *local_dir* to the remote host under *host.remote_base_dir*.

    Creates ``<remote_base_dir>/<local_dir.name>/`` on the remote, then
    uploads every file in *local_dir* (non-recursive).

    Returns
    -------
    str
        Absolute remote path of the uploaded directory.
    """
    _require_paramiko()
    import paramiko  # noqa: PLC0415

    remote_dir = posixpath.join(
        _expand_remote_tilde(host.remote_base_dir, ssh),
        local_dir.name,
    )
    _ssh_run(ssh, f"mkdir -p {remote_dir}")

    sftp = ssh.open_sftp()
    try:
        for local_file in sorted(local_dir.iterdir()):
            if local_file.is_file():
                remote_path = posixpath.join(remote_dir, local_file.name)
                sftp.put(str(local_file), remote_path)
    finally:
        sftp.close()

    return remote_dir


def submit_remote_job(
    script_path: Path,
    host: RemoteHost,
    *,
    ssh: paramiko.SSHClient | None = None,
) -> str:
    """Upload the script's parent directory and submit via ``sbatch``.

    Parameters
    ----------
    script_path : Path
        Local path to the SLURM ``.sh`` script.  The entire parent directory
        is uploaded so that input files are available on the remote.
    host : RemoteHost
        SSH connection details.
    ssh : paramiko.SSHClient or None
        An existing open connection.  If ``None``, a new connection is opened
        and closed automatically.

    Returns
    -------
    str
        SLURM job ID returned by ``sbatch``.
    """
    _require_paramiko()
    _own_conn = ssh is None
    if _own_conn:
        ssh = open_ssh(host)
    try:
        remote_dir = upload_directory(script_path.parent, host, ssh)
        remote_script = posixpath.join(remote_dir, script_path.name)
        stdout, stderr = _ssh_run(ssh, f"cd {remote_dir} && sbatch {remote_script}")
        # sbatch prints "Submitted batch job 12345"
        for line in (stdout + stderr).splitlines():
            parts = line.strip().split()
            if parts and parts[-1].isdigit():
                return parts[-1]
        raise RuntimeError(f"Could not parse job ID from sbatch output:\n{stdout}{stderr}")
    finally:
        if _own_conn:
            ssh.close()


def poll_remote_job(job_id: str, host: RemoteHost, *, ssh=None) -> str:
    """Check the status of a remote SLURM job via ``squeue``.

    Returns one of: ``"queued"``, ``"running"``, ``"done"``, ``"failed"``,
    or ``"unknown"``.
    """
    _require_paramiko()
    _own_conn = ssh is None
    if _own_conn:
        ssh = open_ssh(host)
    try:
        stdout, _ = _ssh_run(
            ssh,
            f"squeue --job {job_id} --format=%T --noheader 2>/dev/null",
        )
        output = stdout.strip().upper()
        if not output:
            return "done"
        if output in ("PENDING", "CF"):
            return "queued"
        if output in ("RUNNING",):
            return "running"
        if output in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY"):
            return "failed"
        return output.lower()
    finally:
        if _own_conn:
            ssh.close()


def fetch_output(
    job_id: str,
    remote_dir: str,
    local_dir: Path,
    host: RemoteHost,
    *,
    ssh=None,
) -> list[Path]:
    """Download all files from *remote_dir* to *local_dir*.

    Useful for retrieving calculation output after a job finishes.

    Returns
    -------
    list[Path]
        Local paths of downloaded files.
    """
    _require_paramiko()
    _own_conn = ssh is None
    if _own_conn:
        ssh = open_ssh(host)
    downloaded: list[Path] = []
    try:
        sftp = ssh.open_sftp()
        try:
            local_dir.mkdir(parents=True, exist_ok=True)
            for entry in sftp.listdir_attr(remote_dir):
                remote_path = posixpath.join(remote_dir, entry.filename)
                local_path = local_dir / entry.filename
                sftp.get(remote_path, str(local_path))
                downloaded.append(local_path)
        finally:
            sftp.close()
    finally:
        if _own_conn:
            ssh.close()
    return downloaded


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ssh_run(ssh, command: str) -> tuple[str, str]:
    """Run *command* on the remote and return (stdout, stderr)."""
    _, stdout_fh, stderr_fh = ssh.exec_command(command)
    stdout = stdout_fh.read().decode()
    stderr = stderr_fh.read().decode()
    return stdout, stderr


def _expand_remote_tilde(path: str, ssh) -> str:
    """Expand a leading ``~`` on the remote host."""
    if not path.startswith("~"):
        return path
    stdout, _ = _ssh_run(ssh, "echo $HOME")
    home = stdout.strip()
    return home + path[1:]


def _require_paramiko() -> None:
    try:
        import paramiko  # noqa: F401
    except ImportError:
        raise ImportError(
            "The 'paramiko' package is required for remote HPC operations.\n"
            "Install it with:  pip install gecko[hpc]  or  pip install paramiko"
        )
