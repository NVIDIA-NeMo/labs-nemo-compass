# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise the real SDK's HTTP parsing and both Langfuse API generations."""

import json
from datetime import datetime, timedelta, timezone
from itertools import count

import httpx
import pytest
from langfuse import Langfuse
from langfuse.api.core.api_error import ApiError
from trace_ingest.loaders.langfuse import (
    LangfuseTraceConfig,
    LangfuseTraceLoader,
    LangfuseTraceLoadError,
)
from trace_ingest.models import UNSET, SpanKind

from insight_agent.cli.main import _configured_trace_loader
from insight_agent.config import LangfuseConfig, TraceConfig

START = datetime(2026, 8, 1, tzinfo=timezone.utc)
END = START + timedelta(hours=1)
CLIENT_IDS = count()


def observation(id, *, trace_id="t", parent=None, start=START, **extra):
    return dict(
        id=id,
        traceId=trace_id,
        projectId="p",
        type="AGENT" if parent is None else "TOOL",
        parentObservationId=parent,
        startTime=start.isoformat(),
        endTime=(start + timedelta(seconds=1)).isoformat(),
        name="lookup",
        level="DEFAULT",
        sessionId="s",
        traceName="assistant",
        **extra,
    )


def page(data, cursor=None):
    return {"data": data, "meta": {"cursor": cursor, "limit": 100}}


def loader(handler, **config):
    client = Langfuse(
        public_key=f"pk-lf-v4-test-{next(CLIENT_IDS)}",
        secret_key="sk-lf-v4-test",
        base_url="https://langfuse.test",
        tracing_enabled=False,
        httpx_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    return LangfuseTraceLoader(
        LangfuseTraceConfig(START, END, base_url="https://langfuse.test", **config), client=client
    )


def test_v4_pagination_full_tree_scores_and_context():
    requests = []
    root = observation("root", start=START, input='{"question":"where?"}')
    child = observation(
        "child", parent="root", start=START + timedelta(seconds=10), output="null", totalCost=0.12
    )

    def handler(request):
        requests.append(request)
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            common = dict(
                projectId="p",
                source="API",
                timestamp=END.isoformat(),
                environment="default",
                createdAt=END.isoformat(),
                updatedAt=END.isoformat(),
                name="quality",
                subject={"kind": "observation", "id": "child", "traceId": "t"},
            )
            if q.get("cursor"):
                return httpx.Response(
                    200,
                    json=page([dict(common, id="score-b", dataType="CATEGORICAL", value="bad")]),
                )
            return httpx.Response(
                200,
                json=page(
                    [dict(common, id="score-a", dataType="BOOLEAN", value=False)], "scores-next"
                ),
            )
        assert request.url.path.endswith("/v2/observations")
        if q.get("limit") == "1":  # Capability probe is independent of filters.
            assert "filter" not in q
            return httpx.Response(200, json=page([]))
        if q.get("traceId"):
            assert q["traceId"] == "t"
            assert "filter" not in q
            assert datetime.fromisoformat(q["fromStartTime"]) == START
            assert datetime.fromisoformat(q["toStartTime"]) == END
            assert "io" in q["fields"] and "trace_context" in q["fields"]
            return httpx.Response(
                200,
                json=page(
                    [root] if q.get("cursor") else [child],
                    None if q.get("cursor") else "details-next",
                ),
            )
        conditions = json.loads(q["filter"])
        assert conditions[-2]["column"] == "startTime"
        assert conditions[-1]["operator"] == "<"
        return httpx.Response(
            200,
            json=page([observation("match")], "selection-next")
            if not q.get("cursor")
            else page([]),
        )

    instance = loader(
        handler,
        filter_string='[{"type":"string","column":"environment","operator":"=","value":"production"}]',
    )
    trace = list(instance.load())[0]
    assert instance.describe()["api_version"] == "v4"
    assert trace.source_url == "https://langfuse.test/project/p/traces/t"
    assert trace.root_spans[0].id == "root"
    tool = trace.root_spans[0].children[0]
    assert tool.kind == SpanKind.TOOL and tool.output is None
    assert trace.root_spans[0].output is UNSET
    assert tool.tool_call.result_count == 1
    assert trace.aggregate.cost_usd == 0.12
    assert trace.aggregate.latency_ms == 11000
    assert trace.attributes["logical_case_id"] == "s"
    assert trace.attributes["langfuse"]["input"] == {"question": "where?"}
    assert [s["value"] for s in trace.evaluator_results["quality"]] == [False, "bad"]
    assert trace.evaluator_results["quality"][0]["subject"]["traceId"] == "t"
    assert len(requests) == 7


@pytest.mark.parametrize("status", [401, 403, 429, 500])
def test_probe_does_not_hide_errors(status):
    requests = []

    def handler(request):
        requests.append(request)
        return httpx.Response(status, json={"message": "failure"})

    with pytest.raises(ApiError):
        loader(handler).load()
    assert all(r.url.path.endswith("/v2/observations") for r in requests)


@pytest.mark.parametrize("mode", ["auto", "v3"])
def test_v3_fallback_and_explicit_override(mode):
    paths = []

    def handler(request):
        paths.append(request.url.path)
        if request.url.path.endswith("/v2/observations"):
            return httpx.Response(404, json={"message": "Not found"})
        assert request.url.path.endswith("/traces")
        return httpx.Response(
            200,
            json={"data": [], "meta": {"page": 1, "limit": 100, "totalItems": 0, "totalPages": 0}},
        )

    instance = loader(handler, api_version=mode)
    assert list(instance.load()) == []
    assert instance.describe()["api_version"] == "v3"
    assert len(paths) == (2 if mode == "auto" else 1)


def test_explicit_v4_does_not_fallback():
    with pytest.raises(ApiError):
        loader(
            lambda r: httpx.Response(404, json={"message": "Not found"}), api_version="v4"
        ).load()


@pytest.mark.parametrize("kind", ["selection", "details", "scores"])
def test_nonprogressing_cursors_fail(kind):
    def handler(request):
        q = request.url.params
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([], "stuck"))
        if q.get("traceId"):
            return httpx.Response(
                200, json=page([observation("root")], "stuck" if kind == "details" else None)
            )
        return httpx.Response(
            200, json=page([observation("root")], "stuck" if kind == "selection" else None)
        )

    with pytest.raises(LangfuseTraceLoadError, match="cursor"):
        loader(handler, api_version="v4").load()


@pytest.mark.parametrize("case", ["duplicate", "cycle", "wrong_trace", "empty"])
def test_malformed_tree_rejected(case):
    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        if request.url.params.get("traceId"):
            data = {
                "duplicate": [observation("root"), observation("root")],
                "cycle": [observation("root", parent="root")],
                "wrong_trace": [observation("root", trace_id="other")],
                "empty": [],
            }[case]
            return httpx.Response(200, json=page(data))
        return httpx.Response(200, json=page([observation("root")]))

    with pytest.raises(LangfuseTraceLoadError):
        loader(handler, api_version="v4").load()


def test_unique_trace_limit_and_empty_result():
    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        if q.get("traceId"):
            return httpx.Response(200, json=page([observation("root", trace_id=q["traceId"])]))
        return httpx.Response(
            200,
            json=page(
                [observation("a"), observation("b"), observation("c", trace_id="other")], "unused"
            ),
        )

    result = list(loader(handler, api_version="v4", max_traces=1).load())
    assert [t.id for t in result] == ["t"]
    assert list(loader(lambda r: httpx.Response(200, json=page([])), api_version="v4").load()) == []


def test_config_wires_api_version():
    source = LangfuseConfig(from_timestamp=START, to_timestamp=END, api_version="v4")
    instance = _configured_trace_loader(TraceConfig(langfuse=source))
    assert isinstance(instance, LangfuseTraceLoader)
    assert instance.config.api_version == "v4"
    with pytest.raises(ValueError):
        LangfuseConfig.model_validate(
            dict(from_timestamp=START, to_timestamp=END, api_version="v5")
        )


def test_v4_cli_runs_evidence(tmp_path, monkeypatch, select_streams):
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import trace_ingest.loaders.langfuse as adapter
    from nooa.unifiedllm import FakeLLMClient

    import insight_agent.cli.main as cli

    tool = observation(
        "tool", parent="root", input='{"query":"missing"}', output='{"error":"not found"}'
    )
    tool.update(level="ERROR", statusMessage="not found")

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        return httpx.Response(200, json=page([tool, observation("root")]))

    client = loader(handler).client
    monkeypatch.setattr(adapter, "_new_langfuse_client", lambda base_url: client)
    compilation = SimpleNamespace(compile_insights=AsyncMock(return_value=[]))
    monkeypatch.setenv("INSIGHT_AGENT_API_KEY", "test-key")
    monkeypatch.setattr(cli, "_build_llm", lambda *a: FakeLLMClient())
    monkeypatch.setattr(cli, "InsightCompilation", lambda llm: compilation)
    assert (
        cli.main(
            [
                "--trace.langfuse.base-url",
                "https://langfuse.test",
                "--trace.langfuse.api-version",
                "v4",
                "--trace.langfuse.from-timestamp",
                START.isoformat(),
                "--trace.langfuse.to-timestamp",
                END.isoformat(),
                "--evidence-streams",
                json.dumps(
                    select_streams(
                        tool_issues={"include_audit_problems": True}, anomaly_and_patterns={}
                    )
                ),
                "--output-path",
                str(tmp_path / "insights.yml"),
            ]
        )
        == cli.EXIT_OK
    )
    evidence = compilation.compile_insights.await_args.args[0]
    assert {item.stream_name for item in evidence} == {"tool-issues", "anomaly-and-patterns"}
    tool_evidence = next(item for item in evidence if item.stream_name == "tool-issues")
    assert tool_evidence.problems


def test_v4_missing_parent_and_conflicting_sessions_remain_visible():
    orphan = observation("orphan", parent="absent")
    other = observation("other")
    other["sessionId"] = "another-session"

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        return httpx.Response(200, json=page([orphan, other]))

    instance = loader(handler, api_version="v4")
    trace = list(instance.load())[0]
    assert instance.describe()["unresolved_parent_count"] == 1
    assert "logical_case_id" not in trace.attributes
    assert len(trace.root_spans) == 2
    assert trace.root_spans[0].attributes["langfuse"]["session_id"] in {"s", "another-session"}


def test_v4_trace_details_include_time_bounds_to_avoid_server_timeout():
    """Self-hosted v4 can time out on trace-ID-only observation reads."""

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        if q.get("traceId") and not (q.get("fromStartTime") and q.get("toStartTime")):
            return httpx.Response(
                422,
                json={"message": "Please narrow your request", "error": "Request timed out"},
            )
        return httpx.Response(200, json=page([observation("root")]))

    assert [trace.id for trace in loader(handler, api_version="v4").load()] == ["t"]


def test_v4_window_excludes_outside_steps_and_records_partial_scope():
    records = [
        observation("root", start=START - timedelta(seconds=1)),
        observation("child", parent="root", start=START),
        observation("later", parent="root", start=END),
    ]

    def handler(request):
        if request.url.path.endswith("/v3/scores"):
            return httpx.Response(200, json=page([]))
        q = request.url.params
        lower = datetime.fromisoformat(q["fromStartTime"])
        upper = datetime.fromisoformat(q["toStartTime"])
        return httpx.Response(
            200,
            json=page(
                [
                    row
                    for row in records
                    if lower <= datetime.fromisoformat(row["startTime"]) < upper
                ]
            ),
        )

    instance = loader(handler, api_version="v4")
    trace = list(instance.load())[0]
    assert [span.id for span in trace.root_spans] == ["child"]
    assert instance.describe()["unresolved_parent_count"] == 1
    assert trace.attributes["observation_window"] == {
        "from_timestamp": START.isoformat(),
        "to_timestamp": END.isoformat(),
        "may_exclude_trace_steps": True,
    }
