"""Tests for job record persistence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gecko.workflow.jobstore import JobRecord, JobStore, load_store, default_store_path


@pytest.fixture()
def store(tmp_path) -> JobStore:
    return JobStore(tmp_path / "jobs.json")


@pytest.fixture()
def sample_record() -> JobRecord:
    return JobRecord(
        job_id="12345",
        mol_name="SO2",
        code="madness",
        script_path="/tmp/calcs/SO2/madness/run_SO2.sh",
    )


class TestJobRecord:
    def test_defaults(self, sample_record):
        assert sample_record.status == "submitted"
        assert sample_record.hostname == ""
        assert sample_record.remote_dir == ""

    def test_mark_updated(self, sample_record):
        sample_record.mark_updated("running")
        assert sample_record.status == "running"
        assert sample_record.updated_at != ""

    def test_timestamps_are_iso(self, sample_record):
        # Should be parseable ISO-8601
        from datetime import datetime
        datetime.fromisoformat(sample_record.submitted_at)
        datetime.fromisoformat(sample_record.updated_at)


class TestJobStore:
    def test_empty_store(self, store):
        assert store.all() == []
        assert store.active() == []

    def test_add_persists(self, store, sample_record):
        store.add(sample_record)
        # Reload from disk
        reloaded = JobStore(store.path)
        assert len(reloaded.all()) == 1
        assert reloaded.all()[0].job_id == "12345"

    def test_get_by_job_id(self, store, sample_record):
        store.add(sample_record)
        found = store.get("12345")
        assert found is not None
        assert found.mol_name == "SO2"

    def test_get_missing_returns_none(self, store):
        assert store.get("99999") is None

    def test_update_status(self, store, sample_record):
        store.add(sample_record)
        store.update("12345", "running")
        reloaded = JobStore(store.path)
        assert reloaded.get("12345").status == "running"

    def test_active_excludes_terminal(self, store):
        r1 = JobRecord(job_id="1", mol_name="H2O", code="madness", script_path="")
        r2 = JobRecord(job_id="2", mol_name="H2O", code="dalton", script_path="")
        store.add(r1)
        store.add(r2)
        store.update("1", "done")
        active = store.active()
        assert len(active) == 1
        assert active[0].job_id == "2"

    def test_all_newest_first(self, store):
        for i in range(3):
            store.add(JobRecord(job_id=str(i), mol_name="X", code="madness", script_path=""))
        ids = [r.job_id for r in store.all()]
        assert ids == ["2", "1", "0"]

    def test_json_file_is_valid(self, store, sample_record):
        store.add(sample_record)
        data = json.loads(store.path.read_text())
        assert isinstance(data, list)
        assert data[0]["job_id"] == "12345"

    def test_multiple_records_roundtrip(self, tmp_path):
        path = tmp_path / "jobs.json"
        s = JobStore(path)
        for i in range(5):
            s.add(JobRecord(job_id=str(i), mol_name=f"mol{i}", code="madness", script_path=""))
        s2 = JobStore(path)
        assert len(s2.all()) == 5


class TestConvenienceFunctions:
    def test_default_store_path(self, tmp_path):
        p = default_store_path(tmp_path)
        assert p == tmp_path / "jobs.json"

    def test_load_store_creates_empty(self, tmp_path):
        store = load_store(tmp_path)
        assert store.all() == []
        # File not created until something is saved
        assert not store.path.exists()

    def test_load_store_reads_existing(self, tmp_path):
        s1 = load_store(tmp_path)
        s1.add(JobRecord(job_id="42", mol_name="N2", code="dalton", script_path=""))
        s2 = load_store(tmp_path)
        assert s2.get("42") is not None
