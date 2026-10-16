# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib

import pytest

from insight_agent.config import TraceConfig
from insight_agent.telemetry import _input_metadata
from trace_ingest.loaders.trace_loaders import TraceDescription
from trace_ingest.models import Trace, TraceSnapshot

WINDOW = {
    "from_timestamp": "2026-09-07T18:27:35Z",
    "to_timestamp": "2026-10-07T18:27:35Z",
}
LIVE_SOURCES = {
    "langfuse": {**WINDOW, "filter": '[{"column":"environment","value":"production"}]'},
    "langsmith": {
        "project": "agent",
        "filter": 'eq(name,"agent")',
        "tree_filter": 'eq(run_type,"llm")',
    },
    "mlflow_experiment": {"experiment": "agent", "filter": "status = 'OK'"},
    "braintrust": {**WINDOW, "project_id": "project"},
    "intake": {
        "base_url": "https://intake.example.com",
        "workspace": "workspace",
        "query": {
            "started_at_gte": WINDOW["from_timestamp"],
            "started_at_lte": WINDOW["to_timestamp"],
        },
    },
}
FILE_SOURCES = {
    "filesystem": {},
    "gym": {},
    "atif": {},
    "mlflow_export": {},
    "langsmith_trace_export_file": {},
    "langfuse_trace_export_file": {"api_version": "v3"},
}


def description(count) -> TraceDescription:
    return {
        "source": "resolved-source",
        "trace_count": count,
        "call_count": 0,
        "distinct_logical_cases": count,
    }


def test_source_coverage():
    assert set(LIVE_SOURCES) | set(FILE_SOURCES) == set(TraceConfig.model_fields) - {"max_traces"}


@pytest.mark.parametrize("source_type,settings", LIVE_SOURCES.items())
def test_live_source_identity(source_type, settings):
    config = TraceConfig.model_validate({source_type: settings, "max_traces": 160})
    traces = [
        Trace(root_spans=[], aggregate={}, id="first"),
        Trace(root_spans=[], aggregate={}, id="second"),
    ]

    class ResolvedDescription(TraceDescription):
        api_url: str
        project_id: str

    resolved: ResolvedDescription = {
        **description(2),
        "api_url": "https://resolved.example.com",
        "project_id": "resolved-project",
    }
    metadata = _input_metadata(config, resolved, TraceSnapshot(traces))
    assert metadata.source_type == source_type
    assert metadata.endpoint == "https://resolved.example.com"
    assert metadata.project_id == "resolved-project"
    assert metadata.trace_count == 2
    assert (
        metadata.normalized_sha256
        == hashlib.sha256("".join(t.model_dump_json() + "\n" for t in traces).encode()).hexdigest()
    )
    assert metadata.snapshot_saved is False
    assert metadata.source_path is None
    assert metadata.truncated is None
    assert metadata.export_trace_count is None
    assert metadata.model_validate_json(metadata.model_dump_json()) == metadata


@pytest.mark.parametrize("source_type,settings", FILE_SOURCES.items())
def test_file_coordinates(tmp_path, monkeypatch, source_type, settings):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "input.jsonl"
    config = TraceConfig.model_validate({source_type: {**settings, "path": "input.jsonl"}})
    snapshot = TraceSnapshot([Trace(root_spans=[], aggregate={}, id="loaded-before-deletion")])
    metadata = _input_metadata(config, description(1), snapshot)
    # Coordinates and normalized identity survive a file disappearing after ingestion.
    assert metadata.source_path == str(path)
    assert metadata.source_type == source_type
    assert metadata.trace_count == 1
    assert len(metadata.normalized_sha256) == 64


@pytest.mark.parametrize(
    "source_type", ["langsmith_trace_export_file", "langfuse_trace_export_file"]
)
def test_export_directory_coordinates(tmp_path, source_type):
    settings = {"path": tmp_path}
    if source_type == "langfuse_trace_export_file":
        settings["api_version"] = "v3"
    metadata = _input_metadata(
        TraceConfig.model_validate({source_type: settings}), description(0), TraceSnapshot([])
    )
    assert metadata.source_path == str(tmp_path)


def test_langfuse_v4_auxiliary_scores_coordinates(tmp_path):
    source = tmp_path / "observations.json"
    scores = tmp_path / "scores.json"
    config = TraceConfig(
        langfuse_trace_export_file={"api_version": "v4", "path": source, "scores_path": scores}
    )

    class VersionDescription(TraceDescription):
        api_version: str

    resolved: VersionDescription = {**description(0), "api_version": "v4"}
    metadata = _input_metadata(config, resolved, TraceSnapshot([]))
    assert metadata.source_path == str(source)
    assert metadata.scores_path == str(scores)
    assert metadata.api_version == "v4"


@pytest.mark.parametrize(
    "source_type,coordinates,expected",
    [
        (
            "langfuse",
            {"base_url": "https://langfuse.example", "api_version": "v4"},
            {"endpoint": "https://langfuse.example", "api_version": "v4"},
        ),
        (
            "langsmith",
            {"api_url": "https://langsmith.example", "project_id": "resolved"},
            {"endpoint": "https://langsmith.example", "project_id": "resolved"},
        ),
        (
            "mlflow_experiment",
            {"tracking_uri": "file:///mlruns", "experiment_id": "42"},
            {"endpoint": "file:///mlruns", "experiment_id": "42"},
        ),
        (
            "braintrust",
            {"api_url": "https://braintrust.example", "experiment_id": "exp"},
            {"endpoint": "https://braintrust.example", "experiment_id": "exp"},
        ),
        (
            "intake",
            {"workspace": "workspace"},
            {"endpoint": "https://intake.example.com", "workspace": "workspace"},
        ),
    ],
)
def test_resolved_platform_coordinates(source_type, coordinates, expected):
    config = TraceConfig.model_validate({source_type: LIVE_SOURCES[source_type]})
    metadata = _input_metadata(config, {**description(0), **coordinates}, TraceSnapshot([]))
    for key, value in expected.items():
        assert getattr(metadata, key) == value


def test_metadata_schema_is_annotated_and_validated():
    from pydantic import ValidationError

    from insight_agent.telemetry import ExecutionMetadata, InputMetadata, _input_metadata_fields

    for model in (InputMetadata, ExecutionMetadata):
        assert all(field.description for field in model.model_fields.values())
        assert all(
            field.get("description") for field in model.model_json_schema()["properties"].values()
        )
    with pytest.raises(ValidationError):
        InputMetadata.model_validate(
            {"source_type": "unknown", "trace_count": 0, "normalized_sha256": "not-a-digest"}
        )
    with pytest.raises(ValidationError):
        InputMetadata.model_validate(
            {"source_type": "filesystem", "trace_count": "2", "normalized_sha256": "a" * 64}
        )
    metadata = InputMetadata(
        source_type="filesystem",
        trace_count=0,
        normalized_sha256="a" * 64,
        source_path="/tmp/input.jsonl",
    )
    assert _input_metadata_fields(metadata) == {
        "compass.input.source_type": "filesystem",
        "compass.input.trace_count": 0,
        "compass.input.normalized_sha256": "a" * 64,
        "compass.input.snapshot_saved": False,
        "compass.input.source_path": "/tmp/input.jsonl",
        "compass.input.details_location": "inputs.input_source",
    }


def test_long_coordinates_remain_in_root_input():
    from insight_agent.telemetry import InputMetadata, RunInput, _input_metadata_fields

    source = InputMetadata(
        source_type="filesystem",
        trace_count=0,
        normalized_sha256="a" * 64,
        source_path="/" + "long-directory/" * 50,
    )
    config = RunInput(trace={"filesystem": {"path": "input.jsonl"}}, input_source=source)
    encoded = config.model_dump_json()
    assert RunInput.model_validate_json(encoded).input_source == source
    fields = _input_metadata_fields(source)
    assert fields["compass.input.details_location"] == "inputs.input_source"
    assert all(not isinstance(value, str) or len(value) < 500 for value in fields.values())


def test_default_loader_limit_is_resolved_without_claiming_source_exhaustion():
    from insight_agent.cli.main import _configured_trace_loader
    from trace_ingest.loaders.langfuse import LANGFUSE_DEFAULT_MAX_TRACES

    config = TraceConfig(langfuse={**WINDOW, "base_url": "https://langfuse.example"})
    loader = _configured_trace_loader(config)
    metadata = _input_metadata(config, loader.describe(), TraceSnapshot([]))
    assert config.max_traces is None
    assert metadata.max_traces == LANGFUSE_DEFAULT_MAX_TRACES
    assert metadata.truncated is None
