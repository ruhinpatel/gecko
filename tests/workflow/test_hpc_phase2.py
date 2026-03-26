"""Tests for JobHandle and updated submit_job / poll_job in hpc.py."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from gecko.workflow.hpc import JobHandle, _poll_local


class TestJobHandle:
    def test_local_not_remote(self):
        h = JobHandle(job_id="42", script_path="/tmp/run.sh")
        assert not h.is_remote

    def test_remote_flag(self):
        h = JobHandle(job_id="42", hostname="hpc.edu", remote_dir="/scratch/x")
        assert h.is_remote

    def test_defaults(self):
        h = JobHandle(job_id="1")
        assert h.hostname == ""
        assert h.remote_dir == ""
        assert h.script_path == ""


class TestPollLocal:
    @pytest.mark.parametrize("raw,expected", [
        ("PENDING\n", "queued"),
        ("CF\n", "queued"),
        ("RUNNING\n", "running"),
        ("", "done"),
        ("FAILED\n", "failed"),
        ("CANCELLED\n", "failed"),
        ("TIMEOUT\n", "failed"),
    ])
    def test_status_mapping(self, raw, expected):
        result = MagicMock(returncode=0, stdout=raw)
        with patch("subprocess.run", return_value=result):
            assert _poll_local("42") == expected

    def test_returns_unknown_on_nonzero(self):
        result = MagicMock(returncode=1, stdout="")
        with patch("subprocess.run", return_value=result):
            assert _poll_local("42") == "unknown"


class TestSubmitJobLocal:
    def test_returns_job_handle_with_id(self, tmp_path):
        script = tmp_path / "run_H2O.sh"
        script.write_text("#!/bin/bash\n")

        result = MagicMock(returncode=0, stdout="Submitted batch job 77777\n", stderr="")
        with patch("subprocess.run", return_value=result):
            from gecko.workflow.hpc import submit_job
            handle = submit_job(script)

        assert handle.job_id == "77777"
        assert not handle.is_remote
        assert handle.script_path == str(script)

    def test_raises_on_sbatch_failure(self, tmp_path):
        script = tmp_path / "run_H2O.sh"
        script.write_text("#!/bin/bash\n")

        result = MagicMock(returncode=1, stdout="", stderr="sbatch: error: ...")
        with patch("subprocess.run", return_value=result):
            from gecko.workflow.hpc import submit_job
            with pytest.raises(RuntimeError, match="sbatch failed"):
                submit_job(script)
