"""Tests for remote SSH module (mocked — no real SSH connection needed)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from gecko.workflow.remote import (
    RemoteHost,
    _expand_remote_tilde,
    _ssh_run,
    _require_paramiko,
)


@pytest.fixture()
def host() -> RemoteHost:
    return RemoteHost(
        hostname="hpc.example.edu",
        username="researcher",
        port=22,
        remote_base_dir="~/gecko_calcs",
    )


@pytest.fixture()
def mock_ssh():
    """A mock paramiko.SSHClient."""
    ssh = MagicMock()
    # exec_command returns (stdin, stdout, stderr) channel objects
    def _exec(cmd):
        stdout = MagicMock()
        stderr = MagicMock()
        stdout.read.return_value = b""
        stderr.read.return_value = b""
        return MagicMock(), stdout, stderr
    ssh.exec_command.side_effect = _exec
    return ssh


class TestRemoteHost:
    def test_default_port(self):
        h = RemoteHost(hostname="example.com", username="user")
        assert h.port == 22

    def test_default_remote_base_dir(self):
        h = RemoteHost(hostname="example.com", username="user")
        assert h.remote_base_dir == "~/gecko_calcs"


class TestRequireParamiko:
    def test_raises_when_missing(self):
        import sys
        # Temporarily hide paramiko if present
        original = sys.modules.get("paramiko")
        sys.modules["paramiko"] = None  # type: ignore[assignment]
        try:
            with pytest.raises(ImportError, match="paramiko"):
                _require_paramiko()
        finally:
            if original is not None:
                sys.modules["paramiko"] = original
            else:
                del sys.modules["paramiko"]


class TestSshRun:
    def test_returns_stdout_stderr(self):
        # Use a fresh mock (not the fixture) so return_value isn't overridden
        ssh = MagicMock()
        stdout_mock = MagicMock()
        stderr_mock = MagicMock()
        stdout_mock.read.return_value = b"hello\n"
        stderr_mock.read.return_value = b"warn\n"
        ssh.exec_command.return_value = (MagicMock(), stdout_mock, stderr_mock)
        out, err = _ssh_run(ssh, "echo hello")
        assert out == "hello\n"
        assert err == "warn\n"

    def test_command_passed_to_exec(self, mock_ssh):
        _ssh_run(mock_ssh, "ls /tmp")
        mock_ssh.exec_command.assert_called_once_with("ls /tmp")


class TestExpandRemoteTilde:
    def test_expands_tilde(self):
        # Fresh mock so the return_value isn't shadowed by fixture side_effect
        ssh = MagicMock()
        stdout_mock = MagicMock()
        stdout_mock.read.return_value = b"/home/researcher\n"
        ssh.exec_command.return_value = (MagicMock(), stdout_mock, MagicMock())
        result = _expand_remote_tilde("~/calcs", ssh)
        assert result == "/home/researcher/calcs"

    def test_no_tilde_unchanged(self, mock_ssh):
        result = _expand_remote_tilde("/absolute/path", mock_ssh)
        assert result == "/absolute/path"
        mock_ssh.exec_command.assert_not_called()


class TestUploadDirectory:
    def test_creates_remote_dir_and_uploads_files(self, tmp_path, host):
        import sys

        # Create local files
        local_dir = tmp_path / "SO2"
        local_dir.mkdir()
        (local_dir / "SO2.in").write_text("input")
        (local_dir / "run_SO2.sh").write_text("#!/bin/bash")

        mock_sftp = MagicMock()
        mock_ssh = MagicMock()
        mock_ssh.open_sftp.return_value = mock_sftp

        # Inject a fake paramiko so the bare `import paramiko` inside the
        # function body doesn't raise ModuleNotFoundError.
        fake_paramiko = MagicMock()
        with patch.dict(sys.modules, {"paramiko": fake_paramiko}), \
             patch("gecko.workflow.remote._require_paramiko"), \
             patch("gecko.workflow.remote._expand_remote_tilde",
                   return_value="/home/researcher/gecko_calcs"), \
             patch("gecko.workflow.remote._ssh_run") as mock_run:

            from gecko.workflow.remote import upload_directory
            remote_dir = upload_directory(local_dir, host, mock_ssh)

        assert remote_dir == "/home/researcher/gecko_calcs/SO2"
        mock_run.assert_called_once()
        assert "mkdir" in mock_run.call_args[0][1]


class TestSubmitRemoteJob:
    def test_parses_job_id_from_sbatch_output(self, tmp_path, host):
        script = tmp_path / "run_SO2.sh"
        script.write_text("#!/bin/bash\n")

        with patch("gecko.workflow.remote._require_paramiko"), \
             patch("gecko.workflow.remote.open_ssh") as mock_open, \
             patch("gecko.workflow.remote.upload_directory", return_value="/remote/SO2"), \
             patch("gecko.workflow.remote._ssh_run") as mock_run:

            mock_run.return_value = ("Submitted batch job 99999\n", "")
            mock_open.return_value.__enter__ = MagicMock()
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            ssh = MagicMock()
            mock_open.return_value = ssh

            from gecko.workflow.remote import submit_remote_job
            job_id = submit_remote_job(script, host, ssh=ssh)

        assert job_id == "99999"


class TestPollRemoteJob:
    @pytest.mark.parametrize("raw,expected", [
        ("PENDING\n", "queued"),
        ("RUNNING\n", "running"),
        ("", "done"),
        ("FAILED\n", "failed"),
        ("CANCELLED\n", "failed"),
    ])
    def test_status_mapping(self, raw, expected, host):
        with patch("gecko.workflow.remote._require_paramiko"), \
             patch("gecko.workflow.remote._ssh_run") as mock_run:
            mock_run.return_value = (raw, "")
            ssh = MagicMock()

            from gecko.workflow.remote import poll_remote_job
            status = poll_remote_job("12345", host, ssh=ssh)

        assert status == expected
