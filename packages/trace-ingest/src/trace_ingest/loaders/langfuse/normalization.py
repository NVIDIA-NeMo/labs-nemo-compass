# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalize Langfuse records for both live and file sources."""

from __future__ import annotations

import heapq
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from pydantic import JsonValue

from trace_ingest.loaders.langfuse.config import LangfuseTraceLoadError, _LangfuseCorpusDescription
from trace_ingest.models import (
    UNSET,
    Span,
    SpanKind,
    ToolCall,
    Trace,
    TraceAggregate,
    TraceSnapshot,
)

if TYPE_CHECKING:
    from langfuse.api.resources.commons.types.observations_view import ObservationsView
    from langfuse.api.resources.commons.types.trace_with_details import TraceWithDetails
    from langfuse.api.resources.commons.types.trace_with_full_details import TraceWithFullDetails


def _describe_corpus(source: str, snapshot: TraceSnapshot) -> _LangfuseCorpusDescription:
    observation_count = call_count = unresolved_parent_count = 0
    pending = [span for trace in snapshot for span in trace.root_spans]
    while pending:
        span = pending.pop()
        observation_count += 1
        call_count += span.kind is SpanKind.TOOL
        pointer = span.attributes.get("source_pointer")
        unresolved_parent_count += (
            isinstance(pointer, dict) and "unresolved_parent_observation_id" in pointer
        )
        pending.extend(span.children)
    return {
        "source": source,
        "trace_count": snapshot.trace_count,
        "call_count": call_count,
        "distinct_logical_cases": len(
            {str(trace.attributes.get("logical_case_id") or trace.id) for trace in snapshot}
        ),
        "observation_count": observation_count,
        "unresolved_parent_count": unresolved_parent_count,
    }


def _trace_id(provider_trace: TraceWithDetails | TraceWithFullDetails) -> str:
    if not provider_trace.id:
        raise LangfuseTraceLoadError("Langfuse returned a trace without an id")
    return str(provider_trace.id)


def _trace_timestamp(provider_trace: TraceWithDetails | TraceWithFullDetails) -> datetime:
    value = provider_trace.timestamp
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse trace {provider_trace.id!r} timestamp must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _tool_catalog(
    observations: Sequence[ObservationsView],
    effective_parents: Mapping[str, str | None],
) -> dict[str, JsonValue] | None:
    """Extract one safely trace-wide catalog from API-visible Langfuse inputs.

    Langfuse normalizes tool definitions from supported instrumentation metadata,
    including ``gen_ai.tool.definitions``, into an observation's top-level
    ``input.tools`` array. GENERATION catalogs describe the tools actually
    offered to the model and therefore take precedence; repeated catalogs must
    agree because the canonical Trace contract cannot represent catalogs that
    change per call. A root AGENT catalog is only a fallback for reduced traces
    with no generation observations.
    """

    root_agent_catalogs: list[dict[str, JsonValue]] = []
    generation_catalogs: list[dict[str, JsonValue] | None] = []
    for observation in observations:
        kind, _ = _map_observation_kind(observation.type)
        if kind not in {SpanKind.AGENT, SpanKind.LLM}:
            continue
        catalog = _tool_catalog_from_input(observation.input)
        observation_id = str(observation.id)
        if kind is SpanKind.LLM:
            generation_catalogs.append(catalog)
        elif catalog is not None and effective_parents[observation_id] is None:
            root_agent_catalogs.append(catalog)

    if generation_catalogs:
        first = generation_catalogs[0]
        if first is None:
            return None
        return first if all(catalog == first for catalog in generation_catalogs[1:]) else None
    if root_agent_catalogs:
        first = root_agent_catalogs[0]
        return first if all(catalog == first for catalog in root_agent_catalogs[1:]) else None
    return None


def _tool_catalog_from_input(value: object) -> dict[str, JsonValue] | None:
    """Convert Langfuse ``input.tools`` definitions to canonical JSON Schemas."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    tools = value.get("tools")
    if not isinstance(tools, list):
        return None

    catalog: dict[str, JsonValue] = {}
    for tool in tools:
        definition = _tool_definition(tool)
        if definition is None:
            continue
        name, schema = definition
        catalog.setdefault(name, schema)
    return catalog if catalog or not tools else None


def _tool_definition(value: object) -> tuple[str, JsonValue] | None:
    """Flatten the tool-definition forms accepted by Langfuse v3 ingestion."""

    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            return None
    if not isinstance(value, Mapping):
        return None
    nested = value.get("function")
    definition = nested if isinstance(nested, Mapping) else value
    name = definition.get("name")
    if not isinstance(name, str) or not name:
        return None

    schema = definition.get("parameters")
    if schema is None:
        schema = definition.get("parameters_json_schema")
    if schema is None:
        schema = definition.get("inputSchema")
    if isinstance(schema, str):
        try:
            schema = json.loads(schema)
        except json.JSONDecodeError:
            schema = None
    return name, _json_value(schema) if isinstance(schema, Mapping) else None


def _canonical_observation_order(
    trace_id: str,
    by_id: Mapping[str, ObservationsView],
    effective_parents: Mapping[str, str | None],
) -> list[ObservationsView]:
    children: dict[str, list[str]] = {observation_id: [] for observation_id in by_id}
    indegree = {observation_id: 0 for observation_id in by_id}
    for observation_id, parent_id in effective_parents.items():
        if parent_id is not None:
            children[parent_id].append(observation_id)
            indegree[observation_id] += 1

    ready: list[tuple[datetime, str]] = []
    for observation_id, degree in indegree.items():
        if degree == 0:
            heapq.heappush(ready, (_started_at(by_id[observation_id]), observation_id))

    ordered: list[ObservationsView] = []
    while ready:
        _, observation_id = heapq.heappop(ready)
        ordered.append(by_id[observation_id])
        for child_id in children[observation_id]:
            indegree[child_id] -= 1
            if indegree[child_id] == 0:
                heapq.heappush(ready, (_started_at(by_id[child_id]), child_id))
    if len(ordered) != len(by_id):
        raise LangfuseTraceLoadError(f"Langfuse trace {trace_id!r} contains a parent cycle")
    return ordered


def _normalize_observation(
    observation: ObservationsView,
    *,
    unresolved_parent_id: str | None,
    base_pointer: dict[str, JsonValue],
) -> Span:
    observation_id = str(observation.id)
    kind, subtype = _map_observation_kind(observation.type)
    started_at = _started_at(observation)
    ended_at = _optional_datetime(observation.end_time)
    pointer: dict[str, JsonValue] = {**base_pointer, "observation_id": observation_id}
    if unresolved_parent_id is not None:
        pointer["unresolved_parent_observation_id"] = unresolved_parent_id

    level = _enum_value(observation.level)
    status_message = str(observation.status_message) if observation.status_message else None
    error = status_message or "Langfuse observation level ERROR" if level == "ERROR" else None
    name = str(observation.name) if observation.name else None
    return Span(
        id=observation_id,
        kind=kind,
        start_time=started_at,
        end_time=ended_at,
        input=_io_value(observation.input),
        output=_io_value(observation.output),
        cost_usd=_observation_cost(observation),
        model=str(observation.model) if observation.model else None,
        tool_name=name if kind is SpanKind.TOOL else None,
        error=error,
        attributes={
            "name": name,
            "subtype": subtype,
            "duration_ms": ((ended_at - started_at).total_seconds() * 1_000 if ended_at else None),
            "source_pointer": pointer,
            "status": level,
            "status_message": status_message,
            "langfuse": {
                "metadata": _json_value(observation.metadata),
                "usage_details": _json_value(observation.usage_details),
                "cost_details": _json_value(observation.cost_details),
                "model_parameters": _json_value(observation.model_parameters),
                "version": observation.version,
                "environment": observation.environment,
                "prompt_id": observation.prompt_id,
                "prompt_name": observation.prompt_name,
                "prompt_version": observation.prompt_version,
            },
        },
    )


def _map_observation_kind(value: object) -> tuple[SpanKind, str | None]:
    provider_type = str(value or "UNKNOWN").upper()
    if provider_type == "GENERATION":
        return SpanKind.LLM, provider_type
    try:
        return SpanKind(provider_type), None
    except ValueError:
        return SpanKind.UNKNOWN, provider_type


def _started_at(observation: ObservationsView) -> datetime:
    value = observation.start_time
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError(
            f"Langfuse observation {observation.id!r} start time must include a timezone"
        )
    return value.astimezone(timezone.utc)


def _optional_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None or value.utcoffset() is None:
        raise LangfuseTraceLoadError("Langfuse observation end time must include a timezone")
    return value.astimezone(timezone.utc)


def _observation_cost(observation: ObservationsView) -> float | None:
    details = observation.cost_details or {}
    cost = details.get("total")
    if cost is None:
        cost = observation.calculated_total_cost
    return float(cost) if cost is not None and cost >= 0 else None


def _trace_cost(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[ObservationsView],
) -> float | None:
    if provider_trace.total_cost is not None and provider_trace.total_cost >= 0:
        return float(provider_trace.total_cost)
    costs = [
        cost for observation in observations if (cost := _observation_cost(observation)) is not None
    ]
    return sum(costs) if costs else None


def _trace_latency_ms(
    provider_trace: TraceWithFullDetails,
    observations: Sequence[ObservationsView],
) -> float | None:
    if provider_trace.latency is not None and provider_trace.latency >= 0:
        return float(provider_trace.latency) * 1_000
    starts = [_started_at(observation) for observation in observations]
    ends = [
        end
        for observation in observations
        if (end := _optional_datetime(observation.end_time)) is not None
    ]
    if not starts or not ends:
        return None
    return (max(ends) - min(starts)).total_seconds() * 1_000


def _io_value(value: object) -> JsonValue | UNSET:
    if value is None:
        # The v3 API does not distinguish a missing value from an explicitly
        # recorded JSON null.
        return UNSET
    if isinstance(value, str):
        try:
            return _json_value(json.loads(value))
        except json.JSONDecodeError:
            pass
    return _json_value(value)


def _enum_value(value: object) -> str | None:
    if value is None:
        return None
    raw = getattr(value, "value", value)
    return str(raw).upper()


def _json_default(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json", by_alias=True)
    legacy_dict = getattr(value, "dict", None)
    if callable(legacy_dict):
        return legacy_dict(by_alias=True)
    return str(value)


def _json_value(value: object) -> JsonValue:
    return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False, default=_json_default))


def _normalize_trace(
    provider_trace: TraceWithFullDetails,
    *,
    source_pointer: dict[str, JsonValue],
    require_complete_tree: bool = False,
) -> Trace:
    trace_id = _trace_id(provider_trace)
    observations = provider_trace.observations
    by_id: dict[str, ObservationsView] = {}
    for observation in observations:
        observation_id = str(observation.id)
        if observation_id in by_id:
            raise LangfuseTraceLoadError(
                f"Langfuse trace {trace_id!r} contains duplicate observation id {observation_id!r}"
            )
        by_id[observation_id] = observation

    effective_parents: dict[str, str | None] = {}
    unresolved: dict[str, str] = {}
    for observation_id, observation in by_id.items():
        parent_id = (
            str(observation.parent_observation_id) if observation.parent_observation_id else None
        )
        if parent_id is not None and parent_id not in by_id:
            if require_complete_tree:
                raise LangfuseTraceLoadError("incomplete observation tree: missing parent")
            unresolved[observation_id] = parent_id
            parent_id = None
        effective_parents[observation_id] = parent_id

    ordered = _canonical_observation_order(trace_id, by_id, effective_parents)
    normalized_by_id: dict[str, Span] = {}
    root_spans: list[Span] = []
    tool_call_index = 0
    for observation in ordered:
        observation_id = str(observation.id)
        normalized = _normalize_observation(
            observation,
            unresolved_parent_id=unresolved.get(observation_id),
            base_pointer=source_pointer,
        )
        if normalized.kind is SpanKind.TOOL:
            normalized.tool_call = ToolCall(
                call_id=observation_id,
                index=tool_call_index,
                result_count=0 if normalized.output is UNSET else 1,
            )
            tool_call_index += 1
        normalized_by_id[observation_id] = normalized
        parent_id = effective_parents[observation_id]
        if parent_id is None:
            root_spans.append(normalized)
        else:
            normalized_by_id[parent_id].children.append(normalized)

    attributes: dict[str, JsonValue] = {
        "source_pointer": source_pointer,
        "langfuse": {
            "trace_name": provider_trace.name,
            "timestamp": _trace_timestamp(provider_trace).isoformat(),
            "input": _json_value(provider_trace.input),
            "output": _json_value(provider_trace.output),
            "user_id": provider_trace.user_id,
            "session_id": provider_trace.session_id,
            "release": provider_trace.release,
            "version": provider_trace.version,
            "environment": provider_trace.environment,
            "tags": _json_value(provider_trace.tags),
            "public": provider_trace.public,
            "metadata": _json_value(provider_trace.metadata),
            "html_path": provider_trace.html_path,
            "scores": _json_value(provider_trace.scores),
        },
    }
    tool_catalog = _tool_catalog(ordered, effective_parents)
    if tool_catalog is not None:
        attributes["tool_catalog"] = tool_catalog
    if provider_trace.session_id:
        attributes["logical_case_id"] = str(provider_trace.session_id)

    # Keep repeated evaluations and observation associations rather than choosing a winner.
    scores_by_name: dict[str, list[JsonValue]] = {}
    for score in sorted(provider_trace.scores, key=lambda score: (score.name, score.id)):
        scores_by_name.setdefault(score.name, []).append(_json_value(score))

    return Trace(
        id=trace_id,
        root_spans=root_spans,
        aggregate=TraceAggregate(
            cost_usd=_trace_cost(provider_trace, ordered),
            latency_ms=_trace_latency_ms(provider_trace, ordered),
        ),
        attributes=attributes,
        evaluator_results={name: scores for name, scores in scores_by_name.items()},
    )
