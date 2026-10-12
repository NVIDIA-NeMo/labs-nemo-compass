# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from insight_agent.insights_generation.trace_queries import TraceQueries
from insight_agent.traces import Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


def test_repetition_preserves_evidence_and_does_not_merge_unknown_inputs():
    calls = [
        Span(
            id=str(i),
            kind=SpanKind.TOOL,
            tool_name="lookup",
            input={"b": 2, "a": 1} if i % 2 else {"a": 1, "b": 2},
            output=None,
            error="failed" if i == 0 else None,
        )
        for i in range(12)
    ]
    calls += [
        Span(id="missing", kind=SpanKind.TOOL, tool_name="lookup"),
        Span(id="different", kind=SpanKind.TOOL, tool_name="lookup", input={"a": 2}),
    ]
    trace = Trace(
        id="t",
        root_spans=[Span(id="root", kind=SpanKind.AGENT, children=calls)],
        aggregate=TraceAggregate(),
    )
    snapshot = TraceSnapshot([trace])
    result = TraceQueries(snapshot).repeated_tool_calls("t")
    assert len(result) == 1
    assert result[0]["count"] == 12
    assert result[0]["examples_truncated"]
    assert len(result[0]["examples"]) == 10
    assert result[0]["examples"][0] == {
        "span_id": "0",
        "error": "failed",
        "output_recorded": True,
        "output_excerpt": "null",
    }
    assert snapshot.get_trace_by_id("t") == trace
    with pytest.raises(ValueError):
        TraceQueries(snapshot).repeated_tool_calls("t", minimum=1)


def test_explicit_null_inputs_match_but_tools_and_traces_stay_separate():
    traces = [
        Trace(
            id=t,
            root_spans=[
                Span(id=str(i), kind=SpanKind.TOOL, tool_name="x", input=None) for i in range(2)
            ],
            aggregate=TraceAggregate(),
        )
        for t in ["a", "b"]
    ]
    query = TraceQueries(TraceSnapshot(traces))
    assert query.repeated_tool_calls("a") == []
    result = query.repeated_tool_calls("b", minimum=2)
    assert result[0]["count"] == 2
    assert result[0]["trace_id"] == "b"
    assert result[0]["examples"][0]["output_recorded"] is False
