"""Tests for SLURM script generation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import qcelemental as qcel

from gecko.workflow.hpc import SlurmConfig, generate_madness_slurm, generate_dalton_slurm, write_madness_slurm, write_dalton_slurm


@pytest.fixture()
def default_config() -> SlurmConfig:
    return SlurmConfig(
        partition="test-partition",
        nodes=2,
        tasks_per_node=4,
        walltime="02:00:00",
        account="myproject",
        madqc_executable="/opt/madqc",
        dalton_executable="/opt/dalton",
    )


class TestGenerateMadnessSlurm:
    def test_contains_sbatch_header(self, default_config):
        script = generate_madness_slurm(Path("H2O.in"), default_config)
        assert "#!/bin/bash" in script
        assert "#SBATCH" in script

    def test_job_name_from_stem(self, default_config):
        script = generate_madness_slurm(Path("SO2_raman.in"), default_config)
        assert "--job-name=SO2_raman" in script

    def test_partition_and_account(self, default_config):
        script = generate_madness_slurm(Path("H2O.in"), default_config)
        assert "--partition=test-partition" in script
        assert "--account=myproject" in script

    def test_nodes_and_tasks(self, default_config):
        script = generate_madness_slurm(Path("H2O.in"), default_config)
        assert "--nodes=2" in script
        assert "--ntasks-per-node=4" in script

    def test_madqc_call(self, default_config):
        script = generate_madness_slurm(Path("H2O.in"), default_config)
        assert "/opt/madqc" in script
        assert "--wf=response" in script
        assert "--input=H2O.in" in script

    def test_mad_num_threads(self, default_config):
        script = generate_madness_slurm(Path("H2O.in"), default_config)
        assert f"MAD_NUM_THREADS={default_config.mad_num_threads}" in script

    def test_no_account_when_empty(self):
        cfg = SlurmConfig(account="")
        script = generate_madness_slurm(Path("H2O.in"), cfg)
        assert "--account" not in script


class TestGenerateDaltonSlurm:
    def test_contains_sbatch_header(self, default_config):
        script = generate_dalton_slurm(Path("H2O.dal"), Path("H2O_aug-cc-pVDZ.mol"), default_config)
        assert "#!/bin/bash" in script

    def test_job_name_prefix(self, default_config):
        script = generate_dalton_slurm(Path("H2O.dal"), Path("H2O.mol"), default_config)
        assert "--job-name=dalton_H2O" in script

    def test_dalton_invocation(self, default_config):
        script = generate_dalton_slurm(Path("H2O.dal"), Path("H2O.mol"), default_config)
        assert "/opt/dalton" in script
        assert "-dal H2O.dal" in script
        assert "-mol H2O.mol" in script

    def test_memory_flag(self, default_config):
        script = generate_dalton_slurm(Path("H2O.dal"), Path("H2O.mol"), default_config)
        assert f"-gb {default_config.dalton_memory_gb}" in script


class TestWriteSlurmScripts:
    def test_write_madness_creates_executable_script(self, tmp_path, default_config):
        in_file = tmp_path / "H2O.in"
        in_file.write_text("# dummy")
        script_path = write_madness_slurm(in_file, default_config)
        assert script_path.exists()
        assert script_path.suffix == ".sh"
        assert oct(script_path.stat().st_mode)[-3:] == "755"

    def test_write_dalton_creates_executable_script(self, tmp_path, default_config):
        dal_file = tmp_path / "H2O.dal"
        mol_file = tmp_path / "H2O.mol"
        dal_file.write_text("# dummy")
        mol_file.write_text("# dummy")
        script_path = write_dalton_slurm(dal_file, mol_file, default_config)
        assert script_path.exists()
        assert script_path.suffix == ".sh"
