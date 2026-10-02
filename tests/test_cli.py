# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from nooa.unifiedllm import FakeLLMClient

import insight_agent.cli.main as cli
from insight_agent.config import EvidenceStreamsConfig, RunConfig
from insight_agent.evidence_streams.evidence_streams import EvidenceStreamResult, Problem
from insight_agent.evidence_streams.registry import EvidenceStreamRegistry
from insight_agent.insight import Insight, load_insights
from insight_agent.insights_generation.config import load_dotenv
from insight_agent.traces import Trace, TraceAggregate, TraceSnapshot


@pytest.fixture
def clean_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(cli.os, "environ", {})
    monkeypatch.setattr(cli, "load_dotenv", lambda: load_dotenv(tmp_path / ".env"))


def test_missing_environment_exits_before_loading_traces(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langsmith:\n    project: production\n"
        "evidence_streams:\n  user_sentiment:\n    litellm:\n"
        "      model: openai/embedding\n      api_key_env: EMBEDDING_API_KEY\n"
    )
    monkeypatch.setenv("LANGSMITH_API_KEY", "")
    monkeypatch.setenv("EMBEDDING_API_KEY", " \t")
    loader = Mock()
    monkeypatch.setattr(cli, "_configured_trace_loader", loader)
    output = tmp_path / "insights.yml"

    assert cli.main(["--config", str(config_path), "--output-path", str(output)]) == 2

    loader.assert_not_called()
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "Missing required environment settings:\n"
        "  INSIGHT_AGENT_API_KEY — API key for the configured model\n"
        "  LANGSMITH_API_KEY — API key for LangSmith\n"
        "  EMBEDDING_API_KEY — API key named by evidence_streams.user_sentiment.litellm.api_key_env\n\n"
        "Set these in .env in your working directory (NAME=value),\n"
        "or export them in your shell, then rerun the command.\n"
        "Setup: https://github.com/NVIDIA-NeMo/labs-trace-intel/blob/main/docs/model-access.md\n"
    )
    assert not output.exists()


def test_langfuse_checks_only_missing_settings_after_cli_overrides(
    clean_environment, monkeypatch, tmp_path, capsys
):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        "trace:\n  langfuse:\n"
        "    from_timestamp: 2026-01-01T00:00:00Z\n"
        "    to_timestamp: 2026-01-02T00:00:00Z\n"
        "evidence_streams:\n  tool_issues: {}\n"
    )
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "private-model-key")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "private-project-key")
    args = ["--config", str(config_path)]
    assert cli.main(args) == 2
    error = capsys.readouterr().err
    assert "LANGFUSE_SECRET_KEY" in error
    assert "LANGFUSE_BASE_URL" in error
    assert "LANGFUSE_PUBLIC_KEY" not in error
    assert "private-" not in error

    args += ["--trace.langfuse.base-url", "https://langfuse.example.com"]
    assert cli.main(args) == 2
    assert "LANGFUSE_BASE_URL" not in capsys.readouterr().err
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "private-secret-key")
    assert cli._check_environment(cli.get_config(args)) == "private-model-key"


def test_environment_accepts_dotenv_and_existing_key_aliases(
    clean_environment, monkeypatch, tmp_path
):
    (tmp_path / ".env").write_text(
        "ANTHROPIC_API_KEY=dotenv-model-key\nLANGCHAIN_API_KEY=dotenv-trace-key\n"
    )
    monkeypatch.setenv("ANTHROPIC_API_KEY", "exported-model-key")
    config = RunConfig(
        trace={"langsmith": {"project": "production"}}, evidence_streams={"tool_issues": {}}
    )

    assert cli._check_environment(config) == "exported-model-key"


@pytest.mark.parametrize(
    "model",
    ["anthropic/claude-sonnet-4-6", "openrouter/anthropic/claude-sonnet-4.6", "openai/test"],
)
@pytest.mark.parametrize("fallback", ["OPENAI_API_BASE", "OPENAI_BASE_URL"])
def test_model_endpoint_does_not_inherit_another_provider_gateway(
    clean_environment, monkeypatch, model, fallback
):
    monkeypatch.setenv(fallback, "https://gateway.example/v1")
    client = Mock()
    monkeypatch.setattr(cli, "CompletionClient", client)
    config = RunConfig(trace={"filesystem": {"path": "traces.jsonl"}}, model=model)

    cli._build_llm(config, "test-key")
    expected = "https://gateway.example/v1" if model.startswith("openai/") else None
    assert client.call_args.kwargs["api_base"] == expected

    monkeypatch.setenv("INSIGHT_AGENT_API_BASE", "https://explicit.example/v1")
    cli._build_llm(config, "test-key")
    assert client.call_args.kwargs["api_base"] == "https://explicit.example/v1"

    config.api_base = "https://configured.example/v1"
    cli._build_llm(config, "test-key")
    assert client.call_args.kwargs["api_base"] == "https://configured.example/v1"


def test_anthropic_parameters_are_translated_to_native_fields(clean_environment, monkeypatch):
    from litellm.utils import get_optional_params

    client = Mock()
    monkeypatch.setattr(cli, "CompletionClient", client)
    config = RunConfig(
        trace={"filesystem": {"path": "traces.jsonl"}},
        model="anthropic/claude-sonnet-4-6",
        max_tokens=8192,
    )
    cli._build_llm(config, "test-key")
    params = dict(client.call_args.kwargs)
    params["model"] = "claude-sonnet-4-6"
    native = get_optional_params(
        **params,
        custom_llm_provider="anthropic",
        tool_choice="auto",
        tools=[
            {
                "type": "function",
                "function": {"name": "ping", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    )
    assert "reasoning_effort" not in native
    assert native["output_config"]["effort"] == "high"
    assert native["tool_choice"] == {"type": "auto"}


def test_cli_preserves_existing_insights_without_synthesizing_empty_evidence(
    clean_environment, tmp_path, monkeypatch, capsys, select_streams
):
    existing = [
        Insight(
            name="Search omits archived documents",
            description="Archived documents disappear from search results.",
            trace_refs=["historical-trace-1", "historical-trace-2"],
        )
    ]
    existing_path = tmp_path / "existing.json"
    existing_path.write_text(json.dumps([item.model_dump() for item in existing]), encoding="utf-8")
    trace_path = tmp_path / "traces.jsonl"
    trace_path.write_text('{"id":"unscored","root_spans":[],"aggregate":{}}', encoding="utf-8")
    output_path = tmp_path / "results" / "insights.yml"
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=existing))
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key-not-real")
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)

    result = cli.main(
        [
            "--trace.filesystem.path",
            str(trace_path),
            "--evidence-streams",
            json.dumps(select_streams(tool_issues={"retry_threshold": 3})),
            "--existing-insights",
            str(existing_path),
            "--output-path",
            str(output_path),
        ]
    )

    compilation.compile_insights.assert_not_awaited()
    assert result == cli.EXIT_OK
    captured = capsys.readouterr()
    assert output_path.read_text(encoding="utf-8") == captured.out
    assert "No new insights produced from 1 trace." in captured.err
    assert "1 existing insight retained." in captured.err
    assert "Skipped" in captured.err
    assert "Tool issues" in captured.err and "No tool calls" in captured.err
    assert f"Saved: {output_path}" in captured.err
    assert load_insights(output_path) == existing


def test_code_validation_filters_problems_and_preserves_stream_result(
    tmp_path, monkeypatch
) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    snapshot = TraceSnapshot([trace])
    supported = Problem(description="Supported", supporting_trace_ids=("trace-1",))
    unsupported = Problem(description="Unsupported", supporting_trace_ids=("trace-1",))
    unknown = Problem(description="Unknown", supporting_trace_ids=("trace-1",))
    evidence = [
        EvidenceStreamResult(
            stream_name="test-stream",
            problems=(supported, unsupported, unknown),
            artifacts={"kept": True},
        )
    ]
    received = []

    class FakeValidator:
        def __init__(self, code_base_path, llm):
            assert code_base_path == tmp_path.resolve()

        async def is_supported(self, problem, supporting_traces):
            received.append((problem, supporting_traces))
            return {"Supported": True, "Unsupported": False, "Unknown": None}[problem.description]

    monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)

    result = asyncio.run(
        cli._validate_evidence_with_code(evidence, snapshot, tmp_path, FakeLLMClient())
    )

    assert result[0].problems == (supported, unknown)
    assert result[0].artifacts == {"kept": True}
    assert received == [
        (supported, (trace,)),
        (unsupported, (trace,)),
        (unknown, (trace,)),
    ]


@pytest.mark.parametrize("concurrency", [None, 2, 8])
def test_code_validation_bounds_concurrency_across_streams_and_preserves_results(
    tmp_path, monkeypatch, concurrency
) -> None:
    args = ["--trace.filesystem.path", "traces.jsonl"]
    if concurrency is not None:
        args.extend(["--code-validation-concurrency", str(concurrency)])
    config = RunConfig(_cli_parse_args=args)
    traces = [
        Trace(id=f"trace-{stream}", root_spans=[], aggregate=TraceAggregate())
        for stream in range(3)
    ]
    evidence = [
        EvidenceStreamResult(
            stream_name=f"stream-{stream}",
            problems=tuple(
                Problem(
                    description=f"Problem {stream}-{index}",
                    supporting_trace_ids=(trace.id,),
                )
                for index in range(5)
            ),
            artifacts={"stream": stream},
            finding_count=8,
            limitations=("Some schemas are missing",),
        )
        for stream, trace in enumerate(traces)
    ]
    decisions = {
        problem.description: (True, False, None)[index % 3]
        for result in evidence
        for index, problem in enumerate(result.problems)
    }
    active = peak = 0
    received = []

    class FakeValidator:
        def __init__(self, code_base_path, llm):
            assert code_base_path == tmp_path.resolve()

        async def is_supported(self, problem, supporting_traces):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            received.append((problem, supporting_traces))
            try:
                await asyncio.sleep(0)
                return decisions[problem.description]
            finally:
                active -= 1

    monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)

    result = asyncio.run(
        cli._validate_evidence_with_code(
            evidence,
            TraceSnapshot(traces),
            tmp_path,
            FakeLLMClient(),
            max_concurrency=config.code_validation_concurrency,
        )
    )

    assert peak == (4 if concurrency is None else concurrency)
    assert active == 0
    assert len(received) == 15
    assert [item.stream_name for item in result] == [item.stream_name for item in evidence]
    for original, filtered in zip(evidence, result, strict=True):
        assert filtered.problems == tuple(
            problem for problem in original.problems if decisions[problem.description] is not False
        )
        assert filtered.model_dump(exclude={"problems"}) == original.model_dump(
            exclude={"problems"}
        )
        assert len(original.problems) == 5
    assert all(
        supporting_traces == (traces[int(problem.description.split()[1].split("-")[0])],)
        for problem, supporting_traces in received
    )


def test_code_validation_propagates_validator_errors(tmp_path, monkeypatch) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    evidence = [
        EvidenceStreamResult(
            stream_name="test-stream",
            problems=tuple(
                Problem(description=f"Problem {index}", supporting_trace_ids=(trace.id,))
                for index in range(8)
            ),
        )
    ]
    active = 0

    class FakeValidator:
        def __init__(self, code_base_path, llm):
            pass

        async def is_supported(self, problem, supporting_traces):
            nonlocal active
            active += 1
            try:
                await asyncio.sleep(0)
                if problem.description == "Problem 0":
                    raise RuntimeError("validation endpoint failed")
                return True
            finally:
                active -= 1

    monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)

    with pytest.raises(RuntimeError, match="validation endpoint failed"):
        asyncio.run(
            cli._validate_evidence_with_code(
                evidence, TraceSnapshot([trace]), tmp_path, FakeLLMClient()
            )
        )

    assert active == 0
    assert len(evidence[0].problems) == 8


@pytest.mark.parametrize("concurrency", [1, 2])
def test_code_validation_cleans_up_siblings_before_returning_error(
    tmp_path, monkeypatch, concurrency
) -> None:
    trace = Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())
    evidence = [
        EvidenceStreamResult(
            stream_name=f"stream-{stream}",
            problems=tuple(
                Problem(
                    description=f"Problem {stream}-{index}",
                    supporting_trace_ids=(trace.id,),
                )
                for index in range(4)
            ),
        )
        for stream in range(2)
    ]
    failure = RuntimeError("validation endpoint failed")
    calls = []
    active = 0

    async def run():
        release = asyncio.Event()

        class FakeValidator:
            def __init__(self, code_base_path, llm):
                pass

            async def is_supported(self, problem, supporting_traces):
                nonlocal active
                calls.append(problem.description)
                active += 1
                try:
                    await asyncio.sleep(0)
                    if problem.description == "Problem 0-0":
                        raise failure
                    await release.wait()
                    return True
                finally:
                    active -= 1

        monkeypatch.setattr(cli, "ProblemValidation", FakeValidator)
        before = asyncio.all_tasks()
        try:
            with pytest.raises(RuntimeError, match="validation endpoint failed") as caught:
                await cli._validate_evidence_with_code(
                    evidence,
                    TraceSnapshot([trace]),
                    tmp_path,
                    FakeLLMClient(),
                    max_concurrency=concurrency,
                )
            assert caught.value is failure
            pending = asyncio.all_tasks() - before
            calls_at_return = tuple(calls)
            for _ in range(3):
                await asyncio.sleep(0)
            assert not pending
            assert active == 0
            assert tuple(calls) == calls_at_return
        finally:
            pending = asyncio.all_tasks() - before
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)

    asyncio.run(run())


@pytest.mark.parametrize("concurrency", [2, 8])
def test_code_validation_receives_configured_concurrency(
    tmp_path, monkeypatch, concurrency
) -> None:
    config = RunConfig(
        trace={"filesystem": {"path": "traces.jsonl"}},
        code_base=tmp_path,
        code_validation_concurrency=concurrency,
    )
    snapshot = TraceSnapshot([])
    result = cli.RunResult(trace_count=0, evidence=[], insights=[])
    validate = AsyncMock(return_value=[])
    monkeypatch.setattr(cli, "_validate_evidence_with_code", validate)
    monkeypatch.setattr(cli, "_build_llm", lambda config, api_key: FakeLLMClient())

    assert (
        asyncio.run(
            cli._compile_evidence(
                config,
                "test-key-not-real",
                snapshot,
                result,
                cli.RunOutput(cli.Console(stderr=True)),
                datetime(2026, 10, 2, tzinfo=timezone.utc),
            )
        )
        == []
    )
    assert validate.await_count == 1
    received = validate.await_args
    assert received is not None
    assert received.kwargs == {"max_concurrency": concurrency}


def test_evidence_streams_share_cli_loop_and_run_concurrently(monkeypatch):
    async def run():
        loop = asyncio.get_running_loop()
        started = asyncio.Event()
        registry = EvidenceStreamRegistry()

        class Stream:
            def __init__(self, name):
                self.name = name

            def check_prerequisites(self, snapshot):
                pass

            async def analyze(self, snapshot):
                assert asyncio.get_running_loop() is loop
                if self.name == "first":
                    await asyncio.wait_for(started.wait(), timeout=1)
                else:
                    started.set()
                return EvidenceStreamResult(stream_name=self.name, problems=())

        for stream in (Stream("first"), Stream("second")):
            registry.register(stream)
        monkeypatch.setattr(cli, "registered_builtin_streams", lambda **kwargs: registry)
        progress = []
        results = await cli._run_evidence_streams(
            EvidenceStreamsConfig(tool_issues={}),
            TraceSnapshot([Trace(id="trace-1", root_spans=[], aggregate=TraceAggregate())]),
            progress=progress.append,
        )
        assert [result.stream_name for result in results] == ["first", "second"]
        assert set().union(*map(set, progress)) == {"first", "second"}
        assert progress[-1] == ()

    asyncio.run(run())


def test_no_candidates_skips_synthesis_and_file_creation(
    clean_environment, tmp_path, monkeypatch, capsys, select_streams
):
    traces = tmp_path / "traces.jsonl"
    traces.write_text('{"id":"trace-1","root_spans":[],"aggregate":{}}')
    output = tmp_path / "insights.yml"
    compilation = Mock()
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(cli, "InsightCompilation", compilation)

    assert (
        cli.main(
            [
                "--trace.filesystem.path",
                str(traces),
                "--evidence-streams",
                json.dumps(select_streams(tool_issues=True)),
                "--output-path",
                str(output),
            ]
        )
        == 0
    )
    compilation.assert_not_called()
    assert not output.exists()
    captured = capsys.readouterr()
    assert captured.out == "[]\n"
    assert "Saved:" not in captured.err
