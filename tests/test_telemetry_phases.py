# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from functools import partial
from io import StringIO
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from nooa.unifiedllm import FakeLLMClient
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from rich.console import Console

import insight_agent.cli.main as cli
from insight_agent.cli.output import RunOutput
from insight_agent.compass_signals.anomaly_and_patterns.signal import (
    AnomalyAndPatternsCompassSignal,
)
from insight_agent.compass_signals.compass_signals import CompassSignalResult, Problem
from insight_agent.config import CompassSignalsConfig, RunConfig
from insight_agent.insight import Insight, TraceEvidence
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.fixture
def captured_spans(monkeypatch):
    # A private provider avoids changing process-wide Nooa instrumentation.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer("phase-test"), config=TraceConfig())
    monkeypatch.setattr(
        cli, "span", partial(tracer.start_as_current_span, openinference_span_kind="chain")
    )
    yield exporter
    provider.shutdown()


def _snapshot():
    return TraceSnapshot(
        [Trace(id=name, root_spans=[], aggregate=TraceAggregate()) for name in ("a", "b")]
    )


def _output(span):
    return json.loads(span.attributes["output.value"])


def test_signal_spans_distinguish_executed_skipped_and_disabled(monkeypatch, captured_spans):
    async def analyze(self, snapshot):
        with cli.span("analysis-operation"):
            await asyncio.sleep(0)
        return CompassSignalResult(signal_name=self.name, problems=(), finding_count=len(snapshot))

    monkeypatch.setattr(AnomalyAndPatternsCompassSignal, "analyze", analyze)
    config = CompassSignalsConfig(
        tool_issues=False, ethos_divergence=False, grader_failure_patterns=False
    )
    with cli.span("compass") as root:
        results = asyncio.run(
            cli._run_compass_signals(config, _snapshot(), lambda: FakeLLMClient([]))
        )

    spans = {span.name: span for span in captured_spans.get_finished_spans()}
    assert set(spans) == {
        "compass",
        "evidence.anomaly-and-patterns",
        "evidence.user-sentiment",
        "analysis-operation",
    }
    for name in ("evidence.anomaly-and-patterns", "evidence.user-sentiment"):
        assert spans[name].parent.span_id == root.get_span_context().span_id
    assert (
        spans["analysis-operation"].parent.span_id
        == spans["evidence.anomaly-and-patterns"].context.span_id
    )
    assert _output(spans["evidence.anomaly-and-patterns"])["finding_count"] == 2
    assert (
        _output(spans["evidence.user-sentiment"])["skip_reason"]
        == "No embedding backend configured"
    )
    assert [item.signal_name for item in results] == ["anomaly-and-patterns", "user-sentiment"]


@pytest.mark.parametrize("retain_problem", [True, False])
def test_pipeline_groups_validation_and_confidence_by_actual_work(
    monkeypatch, tmp_path, captured_spans, retain_problem
):
    snapshot = _snapshot()
    problem = Problem(description="Candidate", supporting_trace_ids=("a", "b"))
    insight = Insight(
        name="Finding",
        description="Supported finding",
        evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
    )

    async def analyze(self, loaded):
        assert loaded is snapshot
        return CompassSignalResult(signal_name=self.name, problems=(problem,))

    async def validate(candidate, supporting):
        assert candidate == problem and len(supporting) == 2
        with cli.span("validation-operation"):
            return retain_problem

    async def compile_insights(evidence, loaded, existing, timestamp):
        assert loaded is snapshot and evidence[0].problems == (problem,)
        with cli.span("compilation-operation"):
            return [insight]

    async def rate(candidate, supporting):
        assert candidate.name == insight.name and len(supporting) == 2
        with cli.span("confidence-operation"):
            return "high"

    monkeypatch.setattr(AnomalyAndPatternsCompassSignal, "analyze", analyze)
    monkeypatch.setattr(cli, "_check_environment", lambda config: "dummy")
    monkeypatch.setattr(cli, "_configured_trace_loader", lambda config: Mock(load=lambda: snapshot))
    monkeypatch.setattr(cli, "record_input_metadata", lambda *args: None)
    monkeypatch.setattr(cli, "_build_llm", lambda *args: FakeLLMClient([]))
    monkeypatch.setattr(
        cli, "ProblemValidation", lambda *args: SimpleNamespace(is_supported=validate)
    )
    monkeypatch.setattr(
        cli,
        "InsightCompilation",
        lambda **kwargs: SimpleNamespace(compile_insights=compile_insights),
    )
    monkeypatch.setattr(cli, "InsightConfidence", lambda *args: SimpleNamespace(rate=rate))
    config = RunConfig(
        trace={"filesystem": {"path": tmp_path / "inputs.jsonl"}},
        code_base=tmp_path,
        compass_signals={
            "tool_issues": False,
            "ethos_divergence": False,
            "grader_failure_patterns": False,
            "user_sentiment": False,
        },
    )
    with cli.span("compass"):
        result = asyncio.run(
            cli._generate_insights(
                config, RunOutput(Console(file=StringIO(), force_terminal=False))
            )
        )

    spans = {span.name: span for span in captured_spans.get_finished_spans()}
    assert _output(spans["ingestion"]) == {"trace_count": 2}
    assert _output(spans["validation"]) == {
        "retained": int(retain_problem),
        "rejected": int(not retain_problem),
    }
    assert spans["validate_problem"].parent.span_id == spans["validation"].context.span_id
    assert spans["validation-operation"].parent.span_id == spans["validate_problem"].context.span_id
    assert _output(spans["validate_problem"]) == {"supported": retain_problem}
    assert result.rejected_by_code == int(not retain_problem)
    if retain_problem:
        assert spans["compilation-operation"].parent.span_id == spans["compilation"].context.span_id
        assert spans["confidence-operation"].parent.span_id == spans["confidence"].context.span_id
        assert _output(spans["confidence"])["insights"][0]["confidence"] == "high"
        assert result.insights[0].confidence == "high"
    else:
        assert "compilation" not in spans and "confidence" not in spans
        assert result.insights == []
    for name in ("ingestion", "validation", "compilation", "confidence"):
        if name in spans:
            assert spans[name].parent.span_id == spans["compass"].context.span_id
    assert len({span.context.trace_id for span in spans.values()}) == 1
