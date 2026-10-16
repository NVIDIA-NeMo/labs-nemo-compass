# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from builtins import ExceptionGroup
from types import SimpleNamespace

import pytest
from nooa.unifiedllm import FakeLLMClient
from openinference.instrumentation import OITracer, TraceConfig
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import insight_agent.cli.main as cli
import insight_agent.telemetry as telemetry
from insight_agent.compass_signals.anomaly_and_patterns.signal import (
    AnomalyAndPatternsCompassSignal,
)
from insight_agent.compass_signals.compass_signals import CompassSignalResult, Problem
from insight_agent.config import CompassSignalsConfig
from insight_agent.insight import Insight, TraceEvidence
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.fixture
def captured_spans(monkeypatch):
    # A private provider avoids changing process-wide Nooa instrumentation.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = OITracer(provider.get_tracer("phase-test"), config=TraceConfig())
    monkeypatch.setattr(telemetry, "tracer", tracer)
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
        "signal.anomaly_and_patterns",
        "signal.user_sentiment",
        "analysis-operation",
    }
    for name in ("signal.anomaly_and_patterns", "signal.user_sentiment"):
        assert spans[name].parent.span_id == root.get_span_context().span_id
    assert (
        spans["analysis-operation"].parent.span_id
        == spans["signal.anomaly_and_patterns"].context.span_id
    )
    assert _output(spans["signal.anomaly_and_patterns"])["status"] == "completed"
    assert (
        _output(spans["signal.user_sentiment"])["skip_reason"] == "No embedding backend configured"
    )
    assert [item.signal_name for item in results] == ["anomaly-and-patterns", "user-sentiment"]


def test_source_code_validation_groups_problems_and_retains_unresolved(
    monkeypatch, tmp_path, captured_spans
):
    from insight_agent.telemetry_models import SourceCodeValidationOutput

    decisions = {"supported": True, "unsupported": False, "unresolved": None}

    async def validate(problem, supporting):
        return decisions[problem.description]

    monkeypatch.setattr(
        cli, "ProblemValidation", lambda *args: SimpleNamespace(is_supported=validate)
    )
    evidence = [
        CompassSignalResult(
            signal_name="tool-issues",
            problems=tuple(
                Problem(description=name, supporting_trace_ids=("a",)) for name in decisions
            ),
        )
    ]
    summary = SourceCodeValidationOutput()
    with cli.span("source_code_validation"):
        retained = asyncio.run(
            cli._validate_evidence_with_code(
                evidence,
                _snapshot(),
                tmp_path,
                FakeLLMClient([]),
                summary=summary,
            )
        )
    assert [problem.description for problem in retained[0].problems] == ["supported", "unresolved"]
    assert summary.model_dump() == {
        name: [{"signal": "tool_issues", "description": name}] for name in decisions
    }
    children = [
        s for s in captured_spans.get_finished_spans() if s.name == "source_code_validation.problem"
    ]
    assert len(children) == 3
    assert {
        json.loads(s.attributes["input.value"])["problem"]["description"]: _output(s)["supported"]
        for s in children
    } == decisions


@pytest.mark.parametrize("phase", ["source_code_validation", "confidence"])
def test_failed_phase_waits_for_sibling_cancellation(monkeypatch, tmp_path, captured_spans, phase):
    completed = []

    async def exercise():
        sibling_started = asyncio.Event()

        async def work(candidate, supporting):
            identity = candidate.description
            with cli.span(f"work.{identity}"):
                if identity == "fails":
                    await sibling_started.wait()
                    raise RuntimeError("failed child")
                sibling_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    await asyncio.sleep(0)
                    completed.append(identity)

        monkeypatch.setattr(
            cli, "ProblemValidation", lambda *args: SimpleNamespace(is_supported=work)
        )
        monkeypatch.setattr(cli, "InsightConfidence", lambda *args: SimpleNamespace(rate=work))
        with cli.span(phase):
            if phase == "source_code_validation":
                evidence = [
                    CompassSignalResult(
                        signal_name=name,
                        problems=(Problem(description=name, supporting_trace_ids=("a", "b")),),
                    )
                    for name in ("fails", "cancelled")
                ]
                await cli._validate_evidence_with_code(
                    evidence, _snapshot(), tmp_path, FakeLLMClient([])
                )
            else:
                insights = [
                    Insight(
                        name=name,
                        description=name,
                        evidence=[TraceEvidence(trace_id="a"), TraceEvidence(trace_id="b")],
                    )
                    for name in ("fails", "cancelled")
                ]
                await cli._rate_confidence_with_code(
                    insights, _snapshot(), tmp_path, FakeLLMClient([])
                )

    with pytest.raises(ExceptionGroup):
        asyncio.run(exercise())
    assert completed == ["cancelled"]
    spans = captured_spans.get_finished_spans()
    root = next(span for span in spans if span.name == phase)
    by_id = {span.context.span_id: span for span in spans}
    for child in spans:
        if child.parent:
            assert child.end_time <= by_id[child.parent.span_id].end_time
    assert not root.status.is_ok
    cancelled = next(span for span in spans if span.name == "work.cancelled")
    assert not cancelled.status.is_ok
    assert any(
        event.name == "exception"
        and "CancelledError" in str(event.attributes.get("exception.type"))
        for event in cancelled.events
    )


def test_signal_inputs_capture_effective_settings_and_consumed_ethos(
    monkeypatch, tmp_path, captured_spans
):
    from insight_agent.compass_signals.ethos_divergence import ethos_divergence_detector as ethos

    ethos_path = tmp_path / "ETHOS.md"
    original_text = "# Purpose\nHelp people draw. Preserve their work.\n"
    ethos_path.write_text(original_text)
    consumed = []

    async def detect(snapshot, llm, text):
        # The exported text must be what the analysis received, even if the file changes.
        consumed.append(text)
        ethos_path.write_text("Changed after ingestion")
        return []

    async def analyze(self, snapshot):
        return CompassSignalResult(signal_name=self.name, problems=())

    monkeypatch.setattr(ethos, "detect_ethos_divergence", detect)
    monkeypatch.setattr(AnomalyAndPatternsCompassSignal, "analyze", analyze)
    config = CompassSignalsConfig(
        anomaly_and_patterns={"minimum_independent_traces": 7},
        tool_issues={"retry_threshold": 5},
        ethos_divergence={"ethos_path": ethos_path},
        grader_failure_patterns={"max_tool_rounds": 3},
    )
    asyncio.run(cli._run_compass_signals(config, _snapshot(), lambda: FakeLLMClient([])))
    spans = {span.name: span for span in captured_spans.get_finished_spans()}
    for key, settings in config:
        name = f"signal.{key}"
        payload = json.loads(spans[name].attributes["input.value"])
        assert payload["trace_count"] == 2
        assert payload["config"] == settings.model_dump(mode="json")
    assert consumed == [original_text]
    assert (
        json.loads(spans["signal.ethos_divergence"].attributes["input.value"])["ethos"]
        == original_text
    )
