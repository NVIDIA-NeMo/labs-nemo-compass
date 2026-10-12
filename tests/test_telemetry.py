# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from unittest.mock import Mock

import pytest
from nooa import Agent, strategy
from nooa.runtime.hooks import get_hooks
from nooa.strategies import CodeActStrategy
from nooa.unifiedllm import CompletionClient, FakeLLMClient, LLMResponse, ToolCall
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import ValidationError

from insight_agent.cli.main import main
from insight_agent.config import RunConfig, TelemetryConfig
from insight_agent.telemetry import capture, span, telemetry_session
from insight_agent.traces import Trace, TraceAggregate


@pytest.fixture
def exporter(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-langsmith-key")
    monkeypatch.setattr(
        "opentelemetry.exporter.otlp.proto.http.trace_exporter.OTLPSpanExporter",
        lambda **kwargs: exporter,
    )
    return exporter


def config(**kwargs):
    return RunConfig(
        trace={"filesystem": {"path": "unused.jsonl"}},
        telemetry={"tier": "full", "project": "test"},
        **kwargs,
    )


def test_requires_explicit_full_content_opt_in():
    with pytest.raises(ValidationError):
        TelemetryConfig.model_validate({"project": "test"})
    with pytest.raises(ValidationError):
        TelemetryConfig.model_validate({"tier": "operational", "project": "test"})


def test_sdk_preserves_parallel_nesting_errors_full_inputs_and_restores_hooks(exporter):
    previous = get_hooks()

    async def child(name, fail=False):
        with span(name, {"items": list(range(30)), "text": "x" * 5000}) as record:
            await asyncio.sleep(0)
            if fail:
                raise ValueError("expected test failure")
            capture(record, "output", {"ok": True})

    async def run():
        await asyncio.gather(child("a"), child("b"), child("c", True), return_exceptions=True)

    with telemetry_session(config(), Mock()):
        asyncio.run(run())
    assert get_hooks() is previous
    records = exporter.get_finished_spans()
    root = next(r for r in records if r.name == "compass")
    for child_record in (r for r in records if r.name != "compass"):
        assert child_record.parent.span_id == root.context.span_id
        assert len(json.loads(child_record.attributes["input.value"])["items"]) == 30
    assert next(r for r in records if r.name == "c").status.is_ok is False
    with span("disabled") as record:
        assert record is None
    assert len(exporter.get_finished_spans()) == 4


class SmallAgent(Agent):
    @strategy(CodeActStrategy())
    async def answer(self) -> int:  # ty: ignore[empty-body] -- Nooa generates this method.
        """Return 7."""
        ...


def test_library_captures_nooa_code_and_litellm_calls(exporter):
    fake = FakeLLMClient(
        [
            LLMResponse(
                raw_response=None,
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="execute_python",
                        arguments=json.dumps({"code": "return_result(7)"}),
                    )
                ],
                finish_reason="tool_calls",
                assistant_message={"role": "assistant", "content": ""},
            )
        ]
    )

    async def run():
        assert await SmallAgent(llm=fake).answer() == 7
        async with CompletionClient(
            model="openai/test", api_key="dummy", mock_response="Hello"
        ) as llm:
            response = await llm.acall([{"role": "user", "content": "Say hello"}])
            assert response.content == "Hello"

    with telemetry_session(config(), Mock()):
        asyncio.run(run())
    records = exporter.get_finished_spans()
    root = next(r for r in records if r.name == "compass")
    assert all(r.context.trace_id == root.context.trace_id for r in records)
    assert any(r.name == "code_execution" for r in records)
    model = next(r for r in records if r.attributes.get("openinference.span.kind") == "LLM")
    assert model.attributes["llm.input_messages.0.message.content"] == "Say hello"
    assert model.attributes["llm.output_messages.0.message.content"] == "Hello"


def test_cli_exports_snapshot_and_zero_insights_then_respects_opt_out(
    tmp_path, monkeypatch, exporter
):
    source = tmp_path / "input.jsonl"
    source.write_text(
        Trace(id="input", root_spans=[], aggregate=TraceAggregate()).model_dump_json() + "\n"
    )
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "dummy")
    args = [
        "--trace.filesystem.path",
        str(source),
        "--evidence-streams.anomaly-and-patterns",
        "false",
        "--evidence-streams.ethos-divergence",
        "false",
        "--evidence-streams.eval-failure-patterns",
        "false",
        "--evidence-streams.user-sentiment",
        "false",
    ]
    assert main([*args, "--telemetry.tier", "full", "--telemetry.project", "test"]) == 0
    records = exporter.get_finished_spans()
    root = next(r for r in records if r.name == "compass")
    assert json.loads(root.attributes["output.value"])["insights"] == []
    trace = next(r for r in records if r.name == "input_trace")
    assert json.loads(trace.attributes["input.value"])["trace"]["id"] == "input"
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    assert main(args) == 0
    assert len(exporter.get_finished_spans()) == len(records)
