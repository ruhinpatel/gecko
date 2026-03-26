"""Workflow utilities for generating and submitting quantum chemistry calculations."""

from gecko.workflow.geometry import fetch_geometry, load_geometry_from_file
from gecko.workflow.params import DFTParams, MoleculeParams, ResponseParams
from gecko.workflow.writers import DaltonInput, MadnessInput
from gecko.workflow.hpc import JobHandle, SlurmConfig, generate_dalton_slurm, generate_madness_slurm, submit_job, poll_job
from gecko.workflow.jobstore import JobRecord, JobStore, load_store
from gecko.workflow.remote import RemoteHost
from gecko.workflow.input_model import (
    Atom,
    DFTSection,
    MadnessInputFile,
    MoleculeSection,
    ResponseSection,
)
from gecko.workflow.input_parser import parse_madness_input, parse_madness_input_file
from gecko.workflow.input_serializer import serialize_madness_input

__all__ = [
    "fetch_geometry",
    "load_geometry_from_file",
    "DFTParams",
    "MoleculeParams",
    "ResponseParams",
    "MadnessInput",
    "DaltonInput",
    "SlurmConfig",
    "JobHandle",
    "JobRecord",
    "JobStore",
    "load_store",
    "RemoteHost",
    "generate_madness_slurm",
    "generate_dalton_slurm",
    "submit_job",
    "poll_job",
    # New input file API
    "Atom",
    "DFTSection",
    "MadnessInputFile",
    "MoleculeSection",
    "ResponseSection",
    "parse_madness_input",
    "parse_madness_input_file",
    "serialize_madness_input",
]
