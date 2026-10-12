# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Read-only evidence queries for the compilation agent."""

import json
from typing import Any

from nooa.skill import Skill

from insight_agent.traces import UNSET, Span, SpanKind, TraceSnapshot


class TraceQueries(Skill):
    """Find observed repetition; repetition alone does not establish a bug."""

    def __init__(self, snapshot: TraceSnapshot) -> None:
        self._snapshot = snapshot

    def repeated_tool_calls(
        self, trace_id: str, minimum: int = 3, limit: int = 20
    ) -> list[dict[str, Any]]:
        """Group identical tool names and recorded JSON inputs within one trace.

        Includes nonconsecutive calls, in source tree order, not proven temporal
        order. Unknown inputs are excluded. Returns at most limit groups and ten
        example outcomes per group; count includes every occurrence. Inspect the
        referenced full spans to decide whether repetition is wasteful or valid
        polling/recovery. Missing error metadata does not establish success.
        """
        if minimum < 2 or not 1 <= limit <= 100:
            raise ValueError("minimum must be >= 2 and limit must be between 1 and 100")
        trace = self._snapshot.get_trace_by_id(trace_id)
        groups: dict[tuple[str, str], list[Span]] = {}
        pending = list(reversed(trace.root_spans))
        while pending:
            span = pending.pop()
            pending.extend(reversed(span.children))
            if span.kind != SpanKind.TOOL or not span.tool_name or span.input is UNSET:
                continue
            key = (span.tool_name, json.dumps(span.input, sort_keys=True, ensure_ascii=False))
            groups.setdefault(key, []).append(span)
        matches: list[dict[str, Any]] = []
        for (name, arguments), spans in groups.items():
            if len(spans) < minimum:
                continue
            matches.append(
                {
                    "trace_id": trace.id,
                    "tool_name": name,
                    "input_excerpt": arguments[:1000],
                    "input_truncated": len(arguments) > 1000,
                    "count": len(spans),
                    "examples_truncated": len(spans) > 10,
                    "examples": [
                        {
                            "span_id": span.id,
                            "error": span.error[:500] if span.error else None,
                            "output_recorded": span.output is not UNSET,
                            "output_excerpt": (
                                json.dumps(span.output, ensure_ascii=False)[:1000]
                                if span.output is not UNSET
                                else None
                            ),
                        }
                        for span in spans[:10]
                    ],
                }
            )
        return sorted(matches, key=lambda item: -item["count"])[:limit]
