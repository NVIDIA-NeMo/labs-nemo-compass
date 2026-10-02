# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from io import StringIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml
from nooa.unifiedllm import FakeLLMClient
from pydantic import ValidationError
from rich.console import Console
from trace_ingest.source_links import http_source_url

from nemo_compass.cli.output import RunOutput, RunResult
from nemo_compass.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from nemo_compass.insight import Insight, load_insights, resolve_trace_links
from nemo_compass.trace_loaders.fs import FSDataLoader
from nemo_compass.traces import Trace, TraceSnapshot


def trace(id, url=None):
    return Trace(id=id, root_spans=[], aggregate={}, source_url=url)


@pytest.mark.parametrize(
    "url",
    [
        "javascript:alert(1)",
        "https://user:secret@host/trace",
        "https://host/\x1b]8;;evil",
        "file://remote-host/trace",
        "relative/path",
    ],
)
def test_trace_links_reject_unusable_urls(url):
    with pytest.raises(ValidationError):
        trace("a", url)
    assert http_source_url(url) is None


def test_jsonl_links_use_source_file_and_count_physical_lines(tmp_path):
    path = tmp_path / "traces #1.jsonl"
    provider_url = "https://provider.test/traces/a"
    path.write_text(
        "\n" + trace("a", provider_url).model_dump_json() + "\n\n" + trace("b").model_dump_json()
    )
    snapshot = FSDataLoader(path).load()
    assert snapshot.get_trace_by_id("a").source_url == path.as_uri() + "#L2"
    assert snapshot.get_trace_by_id("b").source_url == path.as_uri() + "#L4"


def test_links_are_resolved_from_sources_and_existing_artifacts_only(tmp_path):
    old = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [
                {"trace_id": "old", "url": "https://old.test/trace"},
                {"trace_id": "current", "url": "https://stale.test/trace"},
            ],
        }
    )
    generated = Insight.model_validate(
        {
            "name": old.name,
            "description": old.description,
            "evidence": [
                {"trace_id": "old"},
                {"trace_id": "current"},
                {"trace_id": "missing", "url": "https://invented.test/trace"},
                {"trace_id": "no-link"},
            ],
        }
    )
    snapshot = TraceSnapshot([trace("current", "https://current.test/trace"), trace("no-link")])
    [resolved] = resolve_trace_links([generated], snapshot, [old])
    assert [item.trace_id for item in resolved.evidence] == [
        item.trace_id for item in generated.evidence
    ]
    assert {item.trace_id: item.url for item in resolved.evidence if item.url} == {
        "old": "https://old.test/trace",
        "current": "https://current.test/trace",
    }
    path = tmp_path / "insights.yml"
    path.write_text(yaml.safe_dump([resolved.model_dump()]))
    assert load_insights(path) == [resolved]


@pytest.mark.parametrize("terminal", [False, True])
def test_terminal_links_and_plain_logs(terminal, monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    stream = StringIO()
    insight = Insight.model_validate(
        {
            "name": "Issue [literal]",
            "description": "Details",
            "evidence": [
                {"trace_id": "a", "url": "https://provider.test/trace/a"},
                {"trace_id": "b"},
            ],
        }
    )
    RunOutput(Console(file=stream, force_terminal=terminal, width=120)).report(
        RunResult(2, [], [insight]), Path("-")
    )
    rendered = stream.getvalue()
    assert "https://provider.test/trace/a" in rendered
    assert "Issue [literal]" in rendered and "  Trace b" in rendered
    assert ("\x1b]8;" in rendered) == terminal
    if not terminal:
        assert "\x1b" not in rendered


def test_cli_saves_and_prints_loader_links_after_compilation(
    tmp_path, monkeypatch, capsys, select_streams
):
    import nemo_compass.cli.main as cli

    path = tmp_path / "traces.jsonl"
    path.write_text(trace("a").model_dump_json() + "\n" + trace("b").model_dump_json())
    insight = Insight.model_validate(
        {
            "name": "Repeated failure",
            "description": "Details",
            "evidence": [
                {"trace_id": "a", "url": "https://invented.test/trace"},
                {"trace_id": "b"},
            ],
        }
    )
    monkeypatch.setattr(cli, "_check_environment", lambda config: "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(
        cli,
        "_run_evidence_streams",
        AsyncMock(
            return_value=[
                EvidenceStreamResult(
                    stream_name="tool-issues",
                    problems=(Problem(description="Failure", supporting_trace_ids=("a", "b")),),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "InsightCompilation",
        lambda **kwargs: SimpleNamespace(compile_insights=AsyncMock(return_value=[insight])),
    )
    output = tmp_path / "insights.yml"
    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                str(path),
                "--output-path",
                str(output),
                "--evidence-streams",
                json.dumps(select_streams(tool_issues={})),
            ]
        )
        == cli.EXIT_OK
    )
    captured = capsys.readouterr()
    [saved] = load_insights(output)
    assert {item.trace_id: item.url for item in saved.evidence if item.url} == {
        "a": path.as_uri() + "#L1",
        "b": path.as_uri() + "#L2",
    }
    assert yaml.safe_load(captured.out) == [saved.model_dump()]
    assert "invented.test" not in captured.out + captured.err
    assert {item.trace_id: item.url for item in saved.evidence if item.url}["a"] in captured.err


def test_span_resolution_handles_nested_multiple_and_unknown_spans():
    from nemo_compass.traces import Span, SpanKind

    a = trace("a", "https://ui.test/trace")
    a.root_spans = [
        Span(
            id="root",
            kind=SpanKind.AGENT,
            children=[
                Span(
                    id="child",
                    kind=SpanKind.CHAIN,
                    children=[
                        Span(id="leaf", kind=SpanKind.TOOL, source_url="https://ui.test/leaf")
                    ],
                )
            ],
        )
    ]
    b = trace("b")
    b.root_spans = [Span(id="leaf", kind=SpanKind.TOOL, source_url="https://ui.test/other-leaf")]
    insight = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [
                {
                    "trace_id": "a",
                    "spans": [
                        {"span_id": "leaf", "url": "https://invented.test"},
                        {"span_id": "child"},
                        {"span_id": "leaf", "url": "https://invented.test"},
                        {"span_id": "invented"},
                    ],
                },
                {"trace_id": "b", "spans": [{"span_id": "leaf"}]},
            ],
        }
    )
    [resolved] = resolve_trace_links([insight], TraceSnapshot([a, b]))
    assert {
        item.trace_id: [span.span_id for span in item.spans]
        for item in resolved.evidence
        if item.spans
    } == {"a": ["leaf", "child"], "b": ["leaf"]}
    assert {
        item.trace_id: {span.span_id: span.url for span in item.spans if span.url}
        for item in resolved.evidence
        if any(span.url for span in item.spans)
    } == {
        "a": {"leaf": "https://ui.test/leaf"},
        "b": {"leaf": "https://ui.test/other-leaf"},
    }
    assert {item.trace_id: item.url for item in resolved.evidence if item.url} == {
        "a": "https://ui.test/trace"
    }
    assert {
        item.trace_id: {span.span_id: span.url for span in item.spans if span.url}
        for item in insight.evidence
        if any(span.url for span in item.spans)
    }["a"]["leaf"] == "https://invented.test"


def test_historical_span_links_only_survive_for_absent_traces():
    old = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [
                {"trace_id": "a", "spans": [{"span_id": "leaf", "url": "https://old.test/a"}]},
                {"trace_id": "b", "spans": [{"span_id": "leaf", "url": "https://old.test/b"}]},
            ],
        }
    )
    [resolved] = resolve_trace_links([old], TraceSnapshot([trace("a")]), [old])
    assert {
        item.trace_id: [span.span_id for span in item.spans]
        for item in resolved.evidence
        if item.spans
    } == {"b": ["leaf"]}
    assert {
        item.trace_id: {span.span_id: span.url for span in item.spans if span.url}
        for item in resolved.evidence
        if any(span.url for span in item.spans)
    } == {"b": {"leaf": "https://old.test/b"}}


def test_span_fields_are_optional_and_jsonl_does_not_accept_provider_links(tmp_path):
    from nemo_compass.traces import Span, SpanKind

    item = trace("a")
    item.root_spans = [Span(id="leaf", kind=SpanKind.TOOL, source_url="https://supplied.test")]
    path = tmp_path / "traces.jsonl"
    path.write_text(item.model_dump_json())
    loaded = next(iter(FSDataLoader(path).load()))
    assert loaded.root_spans[0].source_url is None
    assert "source_url" not in loaded.root_spans[0].model_dump()
    insight = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [{"trace_id": "a"}, {"trace_id": "b"}],
        }
    )
    assert "span_refs" not in insight.model_dump()
    assert "span_links" not in insight.model_dump()


def test_cli_emits_resolved_span_links(tmp_path, monkeypatch, capsys, select_streams):
    from nemo_compass.cli import main as cli
    from nemo_compass.traces import Span, SpanKind

    a = trace("a", "https://provider.test/trace")
    a.root_spans = [Span(id="tool", kind=SpanKind.TOOL, source_url="https://provider.test/span")]
    snapshot = TraceSnapshot([a, trace("b")])
    insight = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [
                {"trace_id": "a", "spans": [{"span_id": "tool", "url": "https://invented.test"}]},
                {"trace_id": "b"},
            ],
        }
    )
    monkeypatch.setattr(cli, "_check_environment", lambda config: "fake")
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient())
    monkeypatch.setattr(
        cli, "_configured_trace_loader", lambda config: SimpleNamespace(load=lambda: snapshot)
    )
    monkeypatch.setattr(
        cli,
        "_run_evidence_streams",
        AsyncMock(
            return_value=[
                EvidenceStreamResult(
                    stream_name="tool-issues",
                    problems=(Problem(description="Failure", supporting_trace_ids=("a", "b")),),
                )
            ]
        ),
    )
    monkeypatch.setattr(
        cli,
        "InsightCompilation",
        lambda **kwargs: SimpleNamespace(compile_insights=AsyncMock(return_value=[insight])),
    )
    output = tmp_path / "insights.yml"
    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                "unused.jsonl",
                "--output-path",
                str(output),
                "--evidence-streams",
                json.dumps(select_streams(tool_issues={})),
            ]
        )
        == 0
    )
    captured = capsys.readouterr()
    [saved] = load_insights(output)
    assert {
        item.trace_id: [span.span_id for span in item.spans]
        for item in saved.evidence
        if item.spans
    } == {"a": ["tool"]}
    assert {
        item.trace_id: {span.span_id: span.url for span in item.spans if span.url}
        for item in saved.evidence
        if any(span.url for span in item.spans)
    } == {"a": {"tool": "https://provider.test/span"}}
    assert "Span tool — https://provider.test/span" in captured.err
    assert yaml.safe_load(captured.out) == [saved.model_dump()]
    assert "invented.test" not in captured.out + captured.err


@pytest.mark.parametrize(
    "linked_spans",
    [[], ["span-one-long-identifier"], ["span-one-long-identifier", "span-two-long-identifier"]],
)
def test_report_prefers_span_links_and_falls_back_when_any_are_missing(linked_spans):
    refs = ["trace-one-long-identifier", "trace-two-long-identifier"]
    spans = ["span-one-long-identifier", "span-two-long-identifier"]
    insight = Insight.model_validate(
        {
            "name": "Issue",
            "description": "Details",
            "evidence": [
                {
                    "trace_id": refs[0],
                    "url": "https://ui.test/trace-one",
                    "spans": [
                        {
                            "span_id": sid,
                            "url": f"https://ui.test/{sid}" if sid in linked_spans else None,
                        }
                        for sid in spans
                    ],
                },
                {"trace_id": refs[1], "url": "https://ui.test/trace-two"},
            ],
        }
    )
    before = insight.model_dump()
    stream = StringIO()
    RunOutput(Console(file=stream, force_terminal=False, width=200)).report(
        RunResult(2, [], [insight]), Path("-")
    )
    rendered = stream.getvalue()
    assert "Trace trace-on…" in rendered
    assert "    ↳ Span span-one…" in rendered
    assert "    ↳ Span span-two…" in rendered
    assert ("https://ui.test/trace-one" in rendered) == (len(linked_spans) < 2)
    assert "Trace trace-tw… — https://ui.test/trace-two" in rendered
    for sid in spans:
        if sid in linked_spans:
            assert f"https://ui.test/{sid}" in rendered
            assert f"({sid})" not in rendered
        else:
            assert f"Span {sid[:8]}…" in rendered
    assert insight.model_dump() == before


def test_short_ids_distinguish_shared_prefixes():
    from nemo_compass.cli.output import _short_ids

    ids = ["01a0c54c-582d-7270", "01a0c54c-582f-7431", "01a0c54d-27b0", "short"]
    assert _short_ids(ids) == {
        ids[0]: "01a0c54c-582d…",
        ids[1]: "01a0c54c-582f…",
        ids[2]: "01a0c54d…",
        ids[3]: "short",
    }
    assert _short_ids(["abcdefgh", "abcdefghijk"]) == {
        "abcdefgh": "abcdefgh",
        "abcdefghijk": "abcdefghi…",
    }
