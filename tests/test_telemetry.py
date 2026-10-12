# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import subprocess
import sys

import pytest
import yaml

from insight_agent.traces import Trace, TraceAggregate


@pytest.mark.parametrize("enabled", [False, True])
def test_cli_telemetry_in_fresh_process(tmp_path, enabled):
    """Exercise CLI opt-in, library instrumentation, full inputs, and YAML stdout together."""
    source = tmp_path / "input.jsonl"
    source.write_text(
        Trace(
            id="input",
            root_spans=[],
            aggregate=TraceAggregate(),
            attributes={"items": list(range(30)), "text": "x" * 5000},
        ).model_dump_json()
        + "\n"
    )
    result = subprocess.run(
        [sys.executable, __file__, str(source), str(enabled)],
        env={
            **os.environ,
            "INSIGHT_AGENT_API_KEY": "dummy",
            "LANGSMITH_API_KEY": "test-key",
            "LANGSMITH_TRACING": "true",
        },
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert yaml.safe_load(result.stdout) == []


# Nooa owns process-wide instrumentation. Each case gets the same lifecycle as the CLI.
def _run_cli(source, enabled):
    import json
    from unittest.mock import patch

    from nooa import Agent, strategy
    from nooa.strategies import CodeActStrategy
    from nooa.unifiedllm import CompletionClient, FakeLLMClient, LLMResponse, ToolCall
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from insight_agent.cli.main import main
    from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult
    from insight_agent.telemetry import span

    class SmallAgent(Agent):
        @strategy(CodeActStrategy())
        async def answer(self) -> int:  # ty: ignore[empty-body] -- Nooa generates this method.
            """Return 7."""
            ...

    async def analyze(*args):
        fake = FakeLLMClient(
            [
                LLMResponse(
                    raw_response=None,
                    content="",
                    finish_reason="tool_calls",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="execute_python",
                            arguments='{"code":"return_result(7)"}',
                        )
                    ],
                    assistant_message={"role": "assistant", "content": ""},
                )
            ]
        )
        with span("evidence.test") as record:
            assert await SmallAgent(llm=fake).answer() == 7
            async with CompletionClient(
                model="openai/gpt-4o-mini", api_key="dummy", mock_response="Hello"
            ) as llm:
                assert (
                    await llm.acall([{"role": "user", "content": "Say hello"}])
                ).content == "Hello"
            try:
                with span("failed_tool"):
                    raise ValueError("expected test failure")
            except ValueError:
                pass
            result = EvidenceStreamResult(stream_name="test", problems=())
            record.set_output(result)
            return [result]

    exporter = InMemorySpanExporter()
    args = ["--trace.filesystem.path", str(source)]
    if enabled:
        args += ["--telemetry.tier", "full", "--telemetry.project", "test"]
    with patch("insight_agent.telemetry.exporters.otlp", return_value=exporter) as factory:
        with patch("insight_agent.cli.main._run_evidence_streams", analyze):
            assert main(args) == 0
        assert factory.called == enabled
    records = exporter.get_finished_spans()
    if not enabled:
        assert not records
    else:
        assert factory.call_args.args[1]["Langsmith-Project"] == "test"
        root = next(r for r in records if r.name == "compass")
        assert all(r.context.trace_id == root.context.trace_id for r in records)
        ids = {r.context.span_id for r in records}
        assert all(r.parent.span_id in ids for r in records if r.parent)
        assert json.loads(str((root.attributes or {})["output.value"]))["insights"] == []
        source = next(r for r in records if r.name == "input_trace")
        attrs = json.loads(str((source.attributes or {})["input.value"]))["trace"]["attributes"]
        assert attrs == {"items": list(range(30)), "text": "x" * 5000}
        assert any(r.name == "code_execution" for r in records)
        assert not next(r for r in records if r.name == "failed_tool").status.is_ok
        model = next(
            r for r in records if (r.attributes or {}).get("openinference.span.kind") == "LLM"
        )
        assert model.attributes is not None
        assert model.attributes["llm.input_messages.0.message.content"] == "Say hello"
        assert model.attributes["llm.output_messages.0.message.content"] == "Hello"


if __name__ == "__main__":
    _run_cli(sys.argv[1], sys.argv[2] == "True")
