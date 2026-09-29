# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from unittest.mock import Mock

import pytest
import yaml
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError
from trace_ingest.loaders.custom import CustomTraceConfig, CustomTraceLoader, CustomTraceLoadError

from insight_agent.cli import main as cli
from insight_agent.config import TraceConfig

RECORD = {"id": "first", "root_spans": [], "aggregate": {}, "output": None}


def exporter(tmp_path, body, **kwargs):
    script = tmp_path / "export utility.py"
    script.write_text("import sys, json, os, time\nfrom pathlib import Path\n" + body)
    return CustomTraceConfig(
        command=sys.executable,
        args=[str(script), "{output_path}", "literal ; $(not-a-command)"],
        output_path=tmp_path / "output folder" / "traces.jsonl",
        **kwargs,
    )


def test_external_command_exports_and_is_cached(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("CUSTOM_TEST_TOKEN", "inherited")
    config = exporter(
        tmp_path,
        f"""
assert sys.argv[2] == "literal ; $(not-a-command)"
assert os.environ["CUSTOM_TEST_TOKEN"] == "inherited"
print("export progress")
with Path(sys.argv[1]).open("w") as out:
    out.write("\\n" + json.dumps({RECORD!r}) + "\\n")
    out.write(json.dumps({{"id": "second", "root_spans": [], "aggregate": {{}}}}) + "\\n")
""",
    )
    loader = CustomTraceLoader(config)
    snapshot = loader.load()
    assert [trace.id for trace in snapshot] == ["first", "second"]
    assert next(iter(snapshot)).source_url == config.output_path.as_uri() + "#L2"
    assert next(iter(snapshot)).model_dump()["output"] is None
    assert "output" not in snapshot.get_trace_by_id("second").model_dump()
    assert loader.load() is snapshot
    assert loader.describe() == {
        "source": f"custom:{config.output_path}",
        "trace_count": 2,
        "call_count": 0,
        "distinct_logical_cases": 2,
    }
    captured = capfd.readouterr()
    assert captured.out == ""
    assert captured.err.count("export progress") == 1
    assert not list(config.output_path.parent.glob(".trace-export-*"))


@pytest.mark.parametrize(
    "body, message",
    [
        ("sys.exit(7)", "status 7"),
        ("pass", "did not write"),
        ('Path(sys.argv[1]).write_text("")', "contains no traces"),
        ('Path(sys.argv[1]).write_text("broken")', "malformed JSON"),
        ('Path(sys.argv[1]).write_text(\'{"id": "a"}\')', "invalid trace"),
        (
            f'Path(sys.argv[1]).write_text(json.dumps({RECORD!r}) + "\\n" + json.dumps({RECORD!r}))',
            "duplicate",
        ),
    ],
)
def test_failed_export_preserves_existing_output(tmp_path, body, message):
    config = exporter(tmp_path, body)
    config.output_path.parent.mkdir()
    config.output_path.write_text("previous export")
    with pytest.raises(CustomTraceLoadError, match=message):
        CustomTraceLoader(config).load()
    assert config.output_path.read_text() == "previous export"
    assert not list(config.output_path.parent.glob(".trace-export-*"))


def test_timeout(tmp_path):
    config = exporter(tmp_path, "time.sleep(30)", timeout_seconds=0.1)
    with pytest.raises(CustomTraceLoadError, match="timeout_seconds"):
        CustomTraceLoader(config).load()
    assert not config.output_path.exists()


def test_missing_executable(tmp_path):
    config = CustomTraceConfig(
        command="/missing/exporter", args=["{output_path}"], output_path=tmp_path / "out"
    )
    with pytest.raises(CustomTraceLoadError, match="not found or is not executable"):
        CustomTraceLoader(config).load()


@pytest.mark.parametrize(
    "override",
    [
        {"command": " "},
        {"args": []},
        {"args": ["\0{output_path}"]},
        {"args": "--out file"},
        {"timeout_seconds": 0},
        {"extra": True},
    ],
)
def test_invalid_settings(tmp_path, override):
    values = {"command": sys.executable, "args": ["{output_path}"], "output_path": tmp_path / "out"}
    with pytest.raises(ValidationError):
        CustomTraceConfig(**(values | override))


def test_custom_source_is_exclusive_and_limits_belong_to_utility(tmp_path):
    config = exporter(tmp_path, "pass").model_dump()
    with pytest.raises(ValidationError, match="exactly one"):
        TraceConfig(custom=config, filesystem={"path": "other"})
    with pytest.raises(ValidationError, match="trace.custom.args"):
        TraceConfig(custom=config, max_traces=3)


def test_cli_validate_executes_export_without_inference(tmp_path, monkeypatch, capfd):
    config = exporter(tmp_path, f"Path(sys.argv[1]).write_text(json.dumps({RECORD!r}))")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"trace": {"custom": config.model_dump(mode="json")}}))
    inference = Mock(side_effect=AssertionError("must not run inference"))
    monkeypatch.setattr(cli, "_check_environment", inference)
    assert cli.main(["--config", str(path), "--validate-only"]) == cli.EXIT_OK
    inference.assert_not_called()
    assert "Validated 1 traces" in capfd.readouterr().err


def test_cli_reports_exporter_error(tmp_path, capfd):
    config = exporter(tmp_path, "sys.exit(4)")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"trace": {"custom": config.model_dump(mode="json")}}))
    assert cli.main(["--config", str(path), "--validate-only"]) == cli.EXIT_SETUP
    assert "status 4" in capfd.readouterr().err


def test_cli_runs_real_evidence_stream_after_export(tmp_path, monkeypatch, capfd, select_streams):
    config = exporter(tmp_path, f"Path(sys.argv[1]).write_text(json.dumps({RECORD!r}))")
    path = tmp_path / "config.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "trace": {"custom": config.model_dump(mode="json")},
                "evidence_streams": select_streams(tool_issues={}),
                "output_path": "-",
            }
        )
    )
    monkeypatch.setattr(cli, "_check_environment", lambda config: "fake-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    assert cli.main(["--config", str(path)]) == cli.EXIT_OK
    captured = capfd.readouterr()
    assert yaml.safe_load(captured.out) == []
    assert "No tool calls" in captured.err
