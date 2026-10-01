# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import inspect
import json

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from insight_agent.insight import Insight
from insight_agent.insights_generation.insight_compilation import (
    EvidenceCompletion,
    EvidenceDecision,
    InsightCompilation,
)
from insight_agent.traces import UNSET, Span, SpanKind, Trace, TraceAggregate, TraceSnapshot


def trace(trace_id, stage="enrich", output='{"title":"unfinished'):
    return Trace(
        id=trace_id,
        root_spans=[
            Span(
                id=f"{trace_id}-span", kind=SpanKind.LLM, attributes={"stage": stage}, output=output
            )
        ],
        aggregate=TraceAggregate(),
    )


def truncated_enrich(trace):
    span = trace.root_spans[0]
    # A known wrong stage rules out the claim even if output is unavailable.
    if span.attributes.get("stage") != "enrich":
        return {"status": "no_match"}
    if span.output is UNSET:
        return {"status": "unknown"}
    try:
        json.loads(span.output)
    except ValueError:
        return {"status": "match", "witness_span_ids": (span.id,)}
    return {"status": "no_match"}


@pytest.fixture
def completion_case():
    positive = trace("seed")
    other_stage = trace("other-stage", stage="summarize")
    valid_json = trace("valid-json", output='{"title":"complete"}')
    missing = trace("missing", output=UNSET)
    insight = Insight(
        id="saved",
        name="Truncated enrich JSON",
        description="Original claim",
        trace_refs=["seed", "historical"],
    )
    checks = [
        (positive, "match"),
        (other_stage, "no_match"),
        (valid_json, "no_match"),
        (missing, "unknown"),
    ]
    snapshot = TraceSnapshot(
        [
            positive,
            trace("additional"),
            other_stage,
            valid_json,
            missing,
            trace("irrelevant-missing", stage="summarize", output=UNSET),
        ]
    )
    return insight, checks, snapshot


@pytest.mark.parametrize(
    "fault",
    [
        "runtime",
        "shape",
        "foreign_witness",
        "empty_witness",
        "zero_scan",
    ],
)
def test_invalid_completion_retains_citations_and_reports_failure(completion_case, fault):
    insight, checks, snapshot = completion_case

    def predicate(item):
        if fault == "zero_scan" and item.id != "synthetic-positive":
            return {"status": "unknown" if item.id == "missing" else "no_match"}
        if item.id == "additional":
            if fault == "runtime":
                raise RuntimeError("Extraction failed after an earlier match")
            if fault == "shape":
                return EvidenceDecision.model_construct(status="maybe")
            if fault == "foreign_witness":
                return {"status": "match", "witness_span_ids": ("seed-span",)}
            if fault == "empty_witness":
                return {"status": "match"}
        return truncated_enrich(item)

    if fault == "zero_scan":
        checks[0] = (trace("synthetic-positive"), "match")
    completion = EvidenceCompletion(insight=insight, predicate=predicate, checks=checks)
    with pytest.warns(
        UserWarning, match="(?s)Evidence completion unresolved.*no completion additions"
    ):
        result = completion.apply(snapshot)
    assert result == insight


def test_already_complete_claim_gains_no_refs(completion_case):
    insight, checks, snapshot = completion_case
    insight.trace_refs.append("additional")
    completion = EvidenceCompletion(insight=insight, predicate=truncated_enrich, checks=checks)
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        assert completion.apply(snapshot) == insight


def response(name, arguments, call_id):
    return LLMResponse(
        raw_response=None,
        content="",
        tool_calls=[ToolCall(id=call_id, name=name, arguments=json.dumps(arguments))],
        finish_reason="tool_calls",
        assistant_message={},
    )


@pytest.mark.parametrize(
    "invalid_return, diagnostic",
    [
        (None, "Abstention requires"),
        ("EvidenceCompletion(insight=completion.insight)", "Abstention requires"),
        ("completion.model_copy(update={'unresolved_reason': 'partial'})", "no unresolved_reason"),
        (
            "completion.model_copy(update={'predicate': None, 'unresolved_reason': 'cannot check'})",
            "Abstention requires",
        ),
        ("completion.model_copy(update={'checks': []})", "Check a positive"),
        (
            "completion.model_copy(update={'predicate': lambda t: {'status': 'match', "
            "'witness_span_ids': (t.root_spans[0].id,)}})",
            "expected no_match, got match",
        ),
        ("completion.model_copy(update={'predicate': lambda t: 1 / 0})", "division by zero"),
        (
            "completion.model_copy(update={'predicate': lambda t: {'status': 'match', "
            "'witness_span_ids': ('invented',)}})",
            "Witness does not exist",
        ),
    ],
)
def test_return_validation_repairs_handoff_in_live_session(
    completion_case, invalid_return, diagnostic
):
    insight, checks, snapshot = completion_case
    prepare = (
        "import json\nfrom insight_agent.traces import UNSET\n"
        + inspect.getsource(truncated_enrich)
        + "\nchecks = [(trace_snapshot.get_trace_by_id(i), expected) for i, expected in "
        + repr([(t.id, expected) for t, expected in checks])
        + "]\ncompletion = EvidenceCompletion(insight=existing_insights[0], "
        "predicate=truncated_enrich, checks=checks)\noriginal_predicate = completion.predicate"
    )
    invalid = (
        response("return_result", {"result": [{"insight": insight.model_dump()}]}, "invalid")
        if invalid_return is None
        else response("execute_python", {"code": f"return_result([{invalid_return}])"}, "invalid")
    )
    llm = FakeLLMClient(
        [
            response("execute_python", {"code": prepare}, "prepare"),
            invalid,
            response(
                "execute_python",
                {
                    "code": "assert completion.predicate is original_predicate\n"
                    "return_result([completion])"
                },
                "correct",
            ),
        ]
    )
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = asyncio.run(InsightCompilation(llm=llm).compile_insights([], snapshot, [insight]))
    assert result == [
        insight.model_copy(update={"trace_refs": ["seed", "historical", "additional"]})
    ]
    assert llm.call_count == 3
    assert diagnostic in str(llm.last_messages)
    assert "existing live completion objects" in str(llm.last_messages)


def test_explicit_abstention_requires_nonblank_reason(completion_case):
    insight, _, snapshot = completion_case
    valid_reason = "Required intent is not observable"
    llm = FakeLLMClient(
        [
            response(
                "return_result",
                {"result": [{"insight": insight.model_dump(), "unresolved_reason": "   "}]},
                "abstain",
            ),
            response(
                "return_result",
                {"result": [{"insight": insight.model_dump(), "unresolved_reason": valid_reason}]},
                "correct",
            ),
        ]
    )
    with pytest.warns(UserWarning, match=valid_reason):
        assert asyncio.run(
            InsightCompilation(llm=llm).compile_insights([], snapshot, [insight])
        ) == [insight]
    assert llm.call_count == 2
