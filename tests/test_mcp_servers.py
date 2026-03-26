"""Tests for gecko and MADNESS MCP servers.

Verifies that tools are registered, callable, and return expected results.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from gecko.mcp_server import server as gecko_server
from gecko.madness_mcp_server import server as madness_server

_TEST_INPUT = Path(
    "/gpfs/projects/rjh/adrian/development/madness-worktrees/"
    "molresponse-feature-next/src/apps/madqc_v2/"
    "test_molresponse_h2o_alpha_beta_z.in"
)


def _call(server, tool_name: str, args: dict | None = None):
    """Synchronous helper to call an MCP tool."""
    content, meta = asyncio.get_event_loop().run_until_complete(
        server.call_tool(tool_name, args or {})
    )
    return content[0].text


# ---------------------------------------------------------------------------
# Gecko server tests
# ---------------------------------------------------------------------------


class TestGeckoServer:
    def test_tools_registered(self) -> None:
        tools = asyncio.get_event_loop().run_until_complete(gecko_server.list_tools())
        names = {t.name for t in tools}
        assert "parse_input" in names
        assert "list_molecules" in names
        assert "validate_input" in names
        assert "diff_inputs" in names
        assert "create_input" in names
        assert "generate_calc_inputs" in names
        assert len(names) >= 15

    def test_resources_registered(self) -> None:
        resources = asyncio.get_event_loop().run_until_complete(gecko_server.list_resources())
        uris = {str(r.uri) for r in resources}
        assert "gecko://molecules" in uris
        assert "gecko://schema" in uris

    def test_prompts_registered(self) -> None:
        prompts = asyncio.get_event_loop().run_until_complete(gecko_server.list_prompts())
        names = {p.name for p in prompts}
        assert "setup_calculation" in names
        assert "analyze_calculation" in names

    def test_list_molecules(self) -> None:
        result = _call(gecko_server, "list_molecules", {"pattern": "CH3"})
        assert "CH3OH" in result

    def test_get_molecule(self) -> None:
        result = _call(gecko_server, "get_molecule", {"name": "CH3OH"})
        assert "geometry" in result.lower() or "C" in result

    @pytest.mark.skipif(not _TEST_INPUT.exists(), reason="Test fixture not available")
    def test_parse_input(self) -> None:
        result = _call(gecko_server, "parse_input", {"path": str(_TEST_INPUT)})
        data = json.loads(result)
        assert data["dft"]["xc"] == "hf"
        assert data["response"]["kain"] is True
        assert len(data["atoms"]) == 3

    @pytest.mark.skipif(not _TEST_INPUT.exists(), reason="Test fixture not available")
    def test_validate_input(self) -> None:
        result = _call(gecko_server, "validate_input", {"path": str(_TEST_INPUT)})
        assert "Valid" in result

    @pytest.mark.skipif(not _TEST_INPUT.exists(), reason="Test fixture not available")
    def test_get_parameter(self) -> None:
        result = _call(gecko_server, "get_parameter", {"path": str(_TEST_INPUT), "key": "dft.xc"})
        assert result.strip() == "hf"

    @pytest.mark.skipif(not _TEST_INPUT.exists(), reason="Test fixture not available")
    def test_show_input_json(self) -> None:
        result = _call(gecko_server, "show_input", {"path": str(_TEST_INPUT), "format": "json"})
        data = json.loads(result)
        assert "dft" in data

    def test_input_json_schema(self) -> None:
        result = _call(gecko_server, "input_json_schema", {})
        schema = json.loads(result)
        assert "properties" in schema
        assert "dft" in schema["properties"]

    def test_create_input(self, tmp_path: Path) -> None:
        out = str(tmp_path / "test.in")
        result = _call(gecko_server, "create_input", {
            "output_path": out,
            "set_params": ["dft.xc=b3lyp", "dft.k=8"],
        })
        assert "Created" in result
        assert Path(out).exists()

        from gecko.workflow.input_model import MadnessInputFile
        inp = MadnessInputFile.from_file(out)
        assert inp.dft.xc == "b3lyp"
        assert inp.dft.k == 8


# ---------------------------------------------------------------------------
# MADNESS server tests
# ---------------------------------------------------------------------------


class TestMadnessServer:
    def test_tools_registered(self) -> None:
        tools = asyncio.get_event_loop().run_until_complete(madness_server.list_tools())
        names = {t.name for t in tools}
        assert "rebuild" in names
        assert "build_info" in names
        assert "run_madqc" in names
        assert "search_madness_source" in names
        assert "read_madness_file" in names

    def test_resources_registered(self) -> None:
        resources = asyncio.get_event_loop().run_until_complete(madness_server.list_resources())
        uris = {str(r.uri) for r in resources}
        assert "madness://build-info" in uris
        assert "madness://test-inputs" in uris

    def test_build_info(self) -> None:
        result = _call(madness_server, "build_info")
        assert "Source:" in result
        assert "Build:" in result

    def test_list_test_inputs(self) -> None:
        result = _call(madness_server, "list_test_inputs")
        assert "test_molresponse" in result

    def test_read_test_input(self) -> None:
        result = _call(madness_server, "read_test_input", {"name": "test_molresponse_h2o_alpha_beta_z"})
        assert "dft" in result
        assert "response" in result

    def test_search_source(self) -> None:
        result = _call(madness_server, "search_madness_source", {"pattern": "excited.enable"})
        assert "excited.enable" in result
        assert "ResponseParameters" in result

    def test_read_source_file(self) -> None:
        result = _call(madness_server, "read_madness_file", {
            "path": "src/madness/chem/ResponseParameters.hpp",
            "start_line": 1,
            "num_lines": 10,
        })
        assert "ResponseParameters" in result
