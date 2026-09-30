# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json

import pytest
from pydantic import TypeAdapter

from insight_agent.insight import Insight
from insight_agent.insights_generation.insight_compilation import (
    EvidenceCompletion,
    EvidenceDecision,
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
        "broad_rule",
        "missing_as_negative",
        "runtime",
        "shape",
        "foreign_witness",
        "empty_witness",
        "unchecked",
        "zero_scan",
        "unobservable_claim",
    ],
)
def test_invalid_completion_retains_citations_and_reports_failure(completion_case, fault):
    insight, checks, snapshot = completion_case

    def predicate(item):
        if fault == "broad_rule" and item.root_spans[0].output is not UNSET:
            # Matches every corpus positive, but also the close boundary negatives.
            return {"status": "match", "witness_span_ids": (item.root_spans[0].id,)}
        if fault == "missing_as_negative" and item.root_spans[0].output is UNSET:
            return {"status": "no_match"}
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
    completion = EvidenceCompletion(
        insight=insight, predicate=predicate, checks=[] if fault == "unchecked" else checks
    )
    if fault == "unobservable_claim":
        completion.predicate = None
        completion.unresolved_reason = "Required intent is not observable"
    with pytest.warns(
        UserWarning, match="(?s)Evidence completion unresolved.*no completion additions"
    ):
        result = completion.apply(snapshot)
    assert result == insight


def test_checked_scan_adds_only_matches_and_reports_unknowns(completion_case):
    insight, checks, snapshot = completion_case
    completion = EvidenceCompletion(insight=insight, predicate=truncated_enrich, checks=checks)
    # The agent's return tool must accept the internal result's JSON schema,
    # while passing the executable callable through Python.
    assert TypeAdapter(list[EvidenceCompletion]).json_schema()
    assert "predicate" not in completion.model_dump()
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        result = completion.apply(snapshot)
    assert result == insight.model_copy(update={"trace_refs": ["seed", "historical", "additional"]})

    # An already-complete claim gains no refs on a repeated checked scan.
    completion.insight = result
    with pytest.warns(UserWarning, match="1 of 6 traces"):
        assert completion.apply(snapshot) == result
