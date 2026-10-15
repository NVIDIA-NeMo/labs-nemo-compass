# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

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
def test_live_source_selection_and_references(source_type, settings):
    config = TraceConfig.model_validate({source_type: settings, "max_traces": 160})
    traces = [
        Trace(
            root_spans=[],
            aggregate={},
            id="first",
            source_url="https://platform.example.com/traces/first",
            attributes={"input": "large corpus"},
        ),
        Trace(root_spans=[], aggregate={}, id="second"),
    ]
    snapshot = TraceSnapshot(traces)

    class ResolvedDescription(TraceDescription):
        api_url: str
        project_id: str

    resolved: ResolvedDescription = {
        **description(2),
        "api_url": "https://resolved.example.com",
        "project_id": "resolved-project",
    }
    metadata = _input_metadata(config, resolved, snapshot)
    assert metadata["source_type"] == source_type
    assert metadata["selection"] == config.model_dump(mode="json", exclude_none=True)
    assert metadata["description"] == resolved
    assert metadata["trace_count"] == 2
    assert metadata["traces"] == [
        {"id": "first", "source_url": traces[0].source_url},
        {"id": "second", "source_url": None},
    ]
    assert metadata["files"] == []
    assert (
        metadata["normalized_sha256"]
        == hashlib.sha256("".join(t.model_dump_json() + "\n" for t in traces).encode()).hexdigest()
    )
    assert metadata["snapshot_saved"] is False
    json.dumps(metadata)


@pytest.mark.parametrize("source_type,settings", FILE_SOURCES.items())
def test_file_identity_and_line_reference(tmp_path, monkeypatch, source_type, settings):
    monkeypatch.chdir(tmp_path)
    path = tmp_path / "input.jsonl"
    path.write_bytes(b"original source bytes\n")
    config = TraceConfig.model_validate({source_type: {**settings, "path": "input.jsonl"}})
    snapshot = TraceSnapshot(
        [Trace(root_spans=[], aggregate={}, id="source-id", source_url=path.as_uri() + "#L1")]
    )
    metadata = _input_metadata(config, description(1), snapshot)
    assert metadata["files"] == [
        {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    ]
    assert metadata["traces"] == [{"id": "source-id", "source_url": path.as_uri() + "#L1"}]
    assert metadata["source_type"] == source_type
    # Original bytes and normalized input are different identities.
    assert metadata["normalized_sha256"] != metadata["files"][0]["sha256"]


@pytest.mark.parametrize(
    "source_type", ["langsmith_trace_export_file", "langfuse_trace_export_file"]
)
def test_export_directory_fingerprints(tmp_path, source_type):
    (tmp_path / "b.jsonl").write_text("second")
    (tmp_path / "a.jsonl").write_text("first")
    (tmp_path / "notes.txt").write_text("not an input")
    settings = {"path": tmp_path}
    if source_type == "langfuse_trace_export_file":
        settings["api_version"] = "v3"
    metadata = _input_metadata(
        TraceConfig.model_validate({source_type: settings}), description(0), TraceSnapshot([])
    )
    assert metadata["files"] == [
        {
            "path": str(tmp_path / name),
            "sha256": hashlib.sha256((tmp_path / name).read_bytes()).hexdigest(),
        }
        for name in ["a.jsonl", "b.jsonl"]
    ]


def test_langfuse_v4_includes_auxiliary_scores(tmp_path):
    source = tmp_path / "observations.json"
    scores = tmp_path / "scores.json"
    source.write_text("observations")
    scores.write_text("scores")
    config = TraceConfig(
        langfuse_trace_export_file={"api_version": "v4", "path": source, "scores_path": scores}
    )
    metadata = _input_metadata(config, description(0), TraceSnapshot([]))
    assert metadata["files"] == [
        {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
        for p in [source, scores]
    ]


def test_unavailable_file_preserves_loaded_corpus_identity(tmp_path):
    source = tmp_path / "deleted.jsonl"
    config = TraceConfig(filesystem={"path": source})
    snapshot = TraceSnapshot([Trace(root_spans=[], aggregate={}, id="loaded-before-deletion")])
    metadata = _input_metadata(config, description(1), snapshot)
    assert metadata["files"] == [
        {"path": str(source), "sha256": None, "error": "file unavailable after ingestion"}
    ]
    assert metadata["trace_count"] == 1
    assert len(metadata["normalized_sha256"]) == 64
