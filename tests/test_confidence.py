# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

import pytest
from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall

from insight_agent.insight import Insight
from insight_agent.insights_generation import confidence
from insight_agent.insights_generation.confidence import InsightConfidence
from insight_agent.traces import Trace, TraceAggregate


def _response(content: str, tool_calls: list[ToolCall]) -> LLMResponse:
    return LLMResponse(
        raw_response=None,
        content=content,
        tool_calls=tool_calls,
        finish_reason="tool_calls" if tool_calls else "stop",
        assistant_message={"role": "assistant", "content": content},
        reasoning=None,
        usage=None,
    )


def _insight() -> Insight:
    return Insight(
        name="Requests time out too quickly",
        description="The agent's HTTP client times out before slow tools can respond.",
        trace_refs=["t1", "t2"],
    )


def test_insight_confidence_uses_code_before_deciding(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(id="read-1", name="read_file", arguments=json.dumps({"path": "agent.py"}))
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"confidence": "high"}', []),
        ]
    )
    rater = InsightConfidence(tmp_path, llm)
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    result = asyncio.run(rater.rate(_insight(), (trace,)))

    assert result.confidence == "high"
    assert llm.call_count == 2


def test_insight_confidence_forces_decision_after_tool_limit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(confidence, "_MAX_TOOL_ROUNDS", 1)
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(id="read-1", name="read_file", arguments=json.dumps({"path": "agent.py"}))
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"confidence": "low"}', []),
        ]
    )
    rater = InsightConfidence(tmp_path, llm)
    trace = Trace(id="t1", root_spans=[], aggregate=TraceAggregate())

    result = asyncio.run(rater.rate(_insight(), (trace,)))

    assert result.confidence == "low"
    assert llm.call_count == 2


@pytest.mark.parametrize("tool_rounds", [None, 1, 12])
def test_confidence_requires_successful_code_inspection(tmp_path, tool_rounds, monkeypatch) -> None:
    tool_call = ToolCall(
        id="missing", name="read_file", arguments=json.dumps({"path": "missing.py"})
    )
    responses = []
    if tool_rounds is not None:
        monkeypatch.setattr(confidence, "_MAX_TOOL_ROUNDS", tool_rounds)
        responses.extend(_response("", [tool_call]) for _ in range(tool_rounds))
    responses.append(_response('{"confidence": "high"}', []))
    llm = FakeLLMClient(scripted_responses=responses)

    result = asyncio.run(InsightConfidence(tmp_path, llm).rate(_insight(), ()))

    assert result.confidence is None
