# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared read-only codebase investigation loop."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypeVar

from nooa.unifiedllm import Tool, ToolCall, UnifiedLLM, create_tool_from_callable
from pydantic import BaseModel

from insight_agent.insights_generation.codebase import CodebaseTools

_MAX_TOOL_ROUNDS = 12
ResultT = TypeVar("ResultT", bound=BaseModel)


def _structured_result(content: object, output_model: type[ResultT]) -> ResultT:
    if isinstance(content, str):
        return output_model.model_validate_json(content)
    if isinstance(content, output_model):
        return content
    return output_model.model_validate(content)


def _with_output_contract(
    messages: list[dict[str, Any]], output_model: type[ResultT]
) -> list[dict[str, Any]]:
    schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
    instruction = (
        "When you finish investigating, return only one complete JSON object matching this "
        f"schema, with no Markdown or explanation: {schema}"
    )
    prompted_messages = [message.copy() for message in messages]
    if prompted_messages and prompted_messages[0].get("role") == "system":
        prompted_messages[0]["content"] = (
            f"{prompted_messages[0].get('content', '').rstrip()}\n\n{instruction}"
        )
    else:
        prompted_messages.insert(0, {"role": "system", "content": instruction})
    return prompted_messages


def codebase_tools(code_base_path: Path) -> list[Tool]:
    """Read-only tools confined to one codebase."""

    codebase = CodebaseTools(code_base_path)
    return [
        create_tool_from_callable(codebase.list_files),
        create_tool_from_callable(codebase.search_code),
        create_tool_from_callable(codebase.read_file),
    ]


def _execute_tool_call(tools_by_name: dict[str, Tool], tool_call: ToolCall) -> tuple[Any, bool]:
    tool = tools_by_name.get(tool_call.name)
    if tool is None:
        return {"error": f"unknown tool: {tool_call.name}"}, False

    try:
        arguments = json.loads(tool_call.arguments)
        return tool.callable(**arguments), True
    except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
        return {"error": str(exc)}, False


async def investigate(
    llm: UnifiedLLM,
    tools: list[Tool],
    messages: list[dict[str, Any]],
    output_model: type[ResultT],
    final_prompt: str,
    *,
    max_tool_rounds: int = _MAX_TOOL_ROUNDS,
) -> tuple[ResultT, bool]:
    """Return the structured result and whether a repository tool succeeded."""

    tools_by_name = {tool.name: tool for tool in tools}
    messages = _with_output_contract(messages, output_model)
    used_codebase_tool = False
    for _ in range(max_tool_rounds):
        response = await llm.acall(messages, tools=tools, output_model=output_model)
        if not response.tool_calls:
            return _structured_result(response.content, output_model), used_codebase_tool

        messages.append(response.assistant_message)
        for tool_call in response.tool_calls:
            result, succeeded = _execute_tool_call(tools_by_name, tool_call)
            used_codebase_tool |= succeeded
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result, ensure_ascii=False),
                }
            )

    messages.append({"role": "user", "content": final_prompt})
    response = await llm.acall(messages, output_model=output_model)
    return _structured_result(response.content, output_model), used_codebase_tool


__all__ = ["codebase_tools", "investigate"]
