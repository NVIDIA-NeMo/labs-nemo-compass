# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json

from nooa.unifiedllm import FakeLLMClient, LLMResponse, ToolCall
from pydantic import BaseModel

from insight_agent.insights_generation.investigation import CodebaseInvestigation


class _Decision(BaseModel):
    accepted: bool


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


def _messages() -> list[dict[str, str]]:
    return [{"role": "user", "content": "Investigate the timeout."}]


def test_investigation_runs_codebase_tools_and_returns_structured_result(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(
        id="read-1",
        name="read_file",
        arguments=json.dumps({"path": "agent.py"}),
    )
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"accepted": true}', []),
        ]
    )

    result, used_codebase_tool = asyncio.run(
        CodebaseInvestigation(tmp_path, llm).investigate(
            _messages(), _Decision, "Return a decision."
        )
    )

    assert result.accepted is True
    assert used_codebase_tool is True
    assert llm.call_count == 2
    assert "return only one complete JSON object" in llm.last_messages[0]["content"]
    assert '"accepted"' in llm.last_messages[0]["content"]


def test_investigation_adds_output_contract_when_no_system_message(tmp_path) -> None:
    llm = FakeLLMClient(scripted_responses=[_response('{"accepted": true}', [])])

    result, _ = asyncio.run(
        CodebaseInvestigation(tmp_path, llm).investigate(
            _messages(), _Decision, "Return a decision."
        )
    )

    assert result.accepted is True
    assert llm.last_messages[0]["role"] == "system"
    assert "return only one complete JSON object" in llm.last_messages[0]["content"]
    assert llm.last_messages[1] == _messages()[0]


def test_investigation_reports_failed_tool_calls(tmp_path) -> None:
    tool_call = ToolCall(
        id="missing",
        name="read_file",
        arguments=json.dumps({"path": "missing.py"}),
    )
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"accepted": false}', []),
        ]
    )

    result, used_codebase_tool = asyncio.run(
        CodebaseInvestigation(tmp_path, llm).investigate(
            _messages(), _Decision, "Return a decision."
        )
    )

    assert result.accepted is False
    assert used_codebase_tool is False


def test_investigation_forces_structured_result_after_tool_limit(tmp_path) -> None:
    (tmp_path / "agent.py").write_text("TIMEOUT_SECONDS = 1\n", encoding="utf-8")
    tool_call = ToolCall(
        id="read-1",
        name="read_file",
        arguments=json.dumps({"path": "agent.py"}),
    )
    llm = FakeLLMClient(
        scripted_responses=[
            _response("", [tool_call]),
            _response('{"accepted": true}', []),
        ]
    )

    result, used_codebase_tool = asyncio.run(
        CodebaseInvestigation(tmp_path, llm, max_tool_rounds=1).investigate(
            _messages(), _Decision, "Return a decision."
        )
    )

    assert result.accepted is True
    assert used_codebase_tool is True
    assert llm.last_messages[-1] == {"role": "user", "content": "Return a decision."}
    assert llm.call_count == 2
