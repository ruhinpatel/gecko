"""Persistent job tracking for submitted SLURM calculations.

Jobs are stored as a JSON file (``jobs.json``) inside the calculation root
directory so that ``gecko calc status`` can reload them across sessions.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class JobRecord:
    """Metadata for a single submitted SLURM job.

    Parameters
    ----------
    job_id : str
        SLURM job ID (as returned by ``sbatch``).
    mol_name : str
        Molecule identifier (e.g. ``"SO2"``).
    code : str
        ``"madness"`` or ``"dalton"``.
    script_path : str
        Absolute local path of the SLURM ``.sh`` script.
    remote_dir : str
        Remote directory path, or empty string for local submissions.
    hostname : str
        Login-node hostname, or empty string for local submissions.
    status : str
        Last known status: ``"submitted"``, ``"queued"``, ``"running"``,
        ``"done"``, ``"failed"``, or ``"unknown"``.
    submitted_at : str
        ISO-8601 UTC timestamp of submission.
    updated_at : str
        ISO-8601 UTC timestamp of last status poll.
    """

    job_id: str
    mol_name: str
    code: str
    script_path: str
    remote_dir: str = ""
    hostname: str = ""
    status: str = "submitted"
    submitted_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())

    def mark_updated(self, status: str) -> None:
        self.status = status
        self.updated_at = _now_iso()


class JobStore:
    """Load and persist :class:`JobRecord` objects from a JSON file.

    Parameters
    ----------
    path : Path
        Path to the ``jobs.json`` file.  Created on first save if absent.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._records: list[JobRecord] = []
        if self.path.exists():
            self._load()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, record: JobRecord) -> None:
        """Append a new job record and persist immediately."""
        self._records.append(record)
        self.save()

    def all(self) -> list[JobRecord]:
        """Return all records (newest first)."""
        return list(reversed(self._records))

    def get(self, job_id: str) -> JobRecord | None:
        """Return the record for *job_id*, or ``None`` if not found."""
        for r in self._records:
            if r.job_id == job_id:
                return r
        return None

    def update(self, job_id: str, status: str) -> JobRecord | None:
        """Update the status of a job and persist."""
        record = self.get(job_id)
        if record is not None:
            record.mark_updated(status)
            self.save()
        return record

    def active(self) -> list[JobRecord]:
        """Return records whose status is not terminal."""
        terminal = {"done", "failed"}
        return [r for r in self._records if r.status not in terminal]

    def save(self) -> None:
        """Persist all records to disk."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = [asdict(r) for r in self._records]
        self.path.write_text(json.dumps(data, indent=2))

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _load(self) -> None:
        data = json.loads(self.path.read_text())
        self._records = [JobRecord(**item) for item in data]


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def default_store_path(calc_root: Path) -> Path:
    """Return the conventional ``jobs.json`` path for a calc root dir."""
    return Path(calc_root) / "jobs.json"


def load_store(calc_root: Path) -> JobStore:
    """Load (or create) the job store for *calc_root*."""
    return JobStore(default_store_path(calc_root))


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")
