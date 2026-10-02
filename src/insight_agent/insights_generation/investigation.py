# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared read-only codebase investigation loop."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, TypeVar

from nooa import Agent
from nooa.unifiedllm import LLMResponse, Tool, ToolCall, UnifiedLLM, create_tool_from_callable
from pydantic import BaseModel, ValidationError

from insight_agent.insights_generation.codebase import CodebaseTools

_MAX_TOOL_ROUNDS = 12
_MAX_STRUCTURED_RESPONSE_ATTEMPTS = 3
ResultT = TypeVar("ResultT", bound=BaseModel)
logger = logging.getLogger(__name__)


def _structured_result(content: object, output_model: type[ResultT]) -> ResultT:
    if isinstance(content, str):
        return output_model.model_validate_json(content)
    if isinstance(content, output_model):
        return content
    return output_model.model_validate(content)


class CodebaseInvestigation(Agent):
    """Run one structured investigation with confined, read-only repository tools."""

    def __init__(
        self,
        code_base_path: Path,
        llm: UnifiedLLM,
        *,
        max_tool_rounds: int = _MAX_TOOL_ROUNDS,
        max_structured_response_attempts: int = _MAX_STRUCTURED_RESPONSE_ATTEMPTS,
    ) -> None:
        super().__init__(llm=llm)
        codebase = CodebaseTools(code_base_path)
        self._tools: list[Tool] = [
            create_tool_from_callable(codebase.list_files),
            create_tool_from_callable(codebase.search_code),
            create_tool_from_callable(codebase.read_file),
        ]
        self._tools_by_name = {tool.name: tool for tool in self._tools}
        self._max_tool_rounds = max_tool_rounds
        self._max_structured_response_attempts = max_structured_response_attempts

    def _execute_tool_call(self, tool_call: ToolCall) -> tuple[Any, bool]:
        tool = self._tools_by_name.get(tool_call.name)
        if tool is None:
            return {"error": f"unknown tool: {tool_call.name}"}, False

        try:
            arguments = json.loads(tool_call.arguments)
            return tool.callable(**arguments), True
        except (FileNotFoundError, TypeError, ValueError, json.JSONDecodeError) as exc:
            return {"error": str(exc)}, False

    async def _request(
        self,
        messages: list[dict[str, Any]],
        output_model: type[ResultT],
        tools: list[Tool] | None,
    ) -> tuple[LLMResponse | None, ResultT | None]:
        schema = json.dumps(output_model.model_json_schema(), ensure_ascii=False)
        for _ in range(self._max_structured_response_attempts):
            try:
                response = await self.llm.acall(
                    messages,
                    tools=tools,
                    output_model=output_model,
                )
                result = (
                    None
                    if response.tool_calls
                    else _structured_result(response.content, output_model)
                )
                return response, result
            except (json.JSONDecodeError, TypeError, ValidationError):
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response did not match the required structured output. "
                            "Return only one complete JSON object matching this schema, with no "
                            f"Markdown or explanation: {schema}"
                        ),
                    }
                )

        logger.warning(
            "Model failed to return %s after %d structured-response attempts; "
            "treating the investigation as unresolved",
            output_model.__name__,
            self._max_structured_response_attempts,
        )
        return None, None

    async def investigate(
        self,
        messages: list[dict[str, Any]],
        output_model: type[ResultT],
        final_prompt: str,
    ) -> tuple[ResultT | None, bool]:
        """Return the structured result and whether a repository tool succeeded."""

        used_codebase_tool = False
        for _ in range(self._max_tool_rounds):
            response, result = await self._request(messages, output_model, self._tools)
            if response is None:
                return None, used_codebase_tool
            if not response.tool_calls:
                return result, used_codebase_tool

            messages.append(response.assistant_message)
            for tool_call in response.tool_calls:
                result, succeeded = self._execute_tool_call(tool_call)
                used_codebase_tool |= succeeded
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

        messages.append({"role": "user", "content": final_prompt})
        _, result = await self._request(messages, output_model, None)
        return result, used_codebase_tool


__all__ = ["CodebaseInvestigation"]
